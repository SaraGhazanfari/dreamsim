from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from dataset.dataset import TwoAFCDataset
from util.utils import get_preprocess
from torch.utils.data import DataLoader
import os
import yaml
import logging
from training.train import LightningPerceptualModel
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from DISTS_pytorch import DISTS
from dreamsim import PerceptualModel, dreamsim
from tqdm import tqdm
import pickle
import configargparse
from dreamsim import dreamsim
from advertorch.attacks import L2PGDAttack, LinfPGDAttack
from autoattack import AutoAttack
import matplotlib.pyplot as plt

log = logging.getLogger("lightning.pytorch")
log.propagate = False
log.setLevel(logging.ERROR)


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    ## Run options
    parser.add_argument('--seed', type=int, default=1234)

    ## Checkpoint evaluation options
    parser.add_argument('--eval_root', type=str, help="Path to experiment directory containing a checkpoint to "
                                                      "evaluate and the experiment config.yaml.")
    parser.add_argument('--checkpoint_epoch', type=int, help='Epoch number of the checkpoint to evaluate.')
    parser.add_argument('--load_dir', type=str, default="./models", help='path to pretrained ViT checkpoints.')

    ## Baseline evaluation options
    parser.add_argument('--baseline_model', type=str,
                        help='Which ViT model to evaluate. To evaluate an ensemble of models, pass a comma-separated'
                             'list of models. Accepted models: [dino_vits8, dino_vits16, dino_vitb8, dino_vitb16, '
                             'clip_vitb16, clip_vitb32, clip_vitl14, mae_vitb16, mae_vitl16, mae_vith14, '
                             'open_clip_vitb16, open_clip_vitb32, open_clip_vitl14]')
    parser.add_argument('--baseline_feat_type', type=str,
                        help='What type of feature to extract from the model. If evaluating an ensemble, pass a '
                             'comma-separated list of features (same length as model_type). Accepted feature types: '
                             '[cls, embedding, last_layer].')
    parser.add_argument('--baseline_stride', type=str,
                        help='Stride of first convolution layer the model (should match patch size). If finetuning'
                             'an ensemble, pass a comma-separated list (same length as model_type).')
    parser.add_argument('--baseline_output_path', type=str, help='Path to save evaluation results.')

    ## Dataset options
    parser.add_argument('--nights_root', type=str, default='./dataset/nights', help='path to nights dataset.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4, help='dataset batch size.')
    parser.add_argument('--attack_type', type=str, default=None, help='PGD-L2, PGD-Linf, AA-L2, AA-Linf')
    parser.add_argument('--eps', type=float, default=0, help='epsilon for attack')
    return parser.parse_args()


def model_wrapper(model):
    def metric_model(img):
        img_ref, img_0, img_1 = img[:, 0, :, :].squeeze(1), img[:, 1, :, :].squeeze(1), img[:, 2, :, :].squeeze(1)
        dist_0 = model(img_ref, img_0)
        dist_1 = model(img_ref, img_1)
        return torch.stack((dist_1, dist_0), dim=1)

    return metric_model


def generate_attack(attack_type, model, img_ref, img_0, img_1, target, epsilon):
    attack_method, attack_norm = attack_type.split('-')

    if attack_method == 'AA':
        adversary = AutoAttack(model_wrapper(model), norm=attack_norm, eps=epsilon, version='standard', device=device)
        adversary.attacks_to_run = ['apgd-ce']
        img_ref = adversary.run_standard_evaluation(torch.stack((img_ref, img_0, img_1), dim=1), target.long(),
                                                    bs=img_ref.shape[0])
        img_ref, img_0, img_1 = img_ref[:, 0, :, :].squeeze(1), img_ref[:, 1, :, :].squeeze(1), img_ref[:, 2, :,
                                                                                                :].squeeze(1)
    elif attack_method == 'PGD':
        if attack_norm == 'L2':
            adversary = L2PGDAttack(model.embed, loss_fn=nn.MSELoss(), eps=epsilon, nb_iter=200, rand_init=True,
                                    targeted=False, eps_iter=0.01, clip_min=0.0, clip_max=1.0)

        else:  # attack_type == 'Linf':
            adversary = LinfPGDAttack(model.embed, loss_fn=nn.MSELoss(), eps=epsilon, nb_iter=50,
                                      eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., targeted=False)

        img_ref = adversary(img_ref, model.embed(img_ref))

    return img_ref, img_0, img_1


def calculate_twoafc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores, dim=0)
    logging.info(f"2AFC score: {str(twoafc_score)}")
    return twoafc_score


def score_nights_dataset(model, test_loader, device, attack_type, epsilon=0):
    logging.info("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []

    # with torch.no_grad():
    for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img_ref, img_left, img_right, target = img_ref.to(device), img_left.to(device), \
            img_right.to(device), target.to(device)
        img_left = img_left.detach()
        img_right = img_right.detach()

        if attack_type:
            adv_img_ref, _, _ = generate_attack(attack_type=attack_type, model=model, img_ref=img_ref, img_0=img_left,
                                            img_1=img_right, target=target, epsilon=epsilon)
        # dist_0 = model(img_ref, img_left)
        # dist_1 = model(img_ref, img_right)
        #
        # if len(dist_0.shape) < 1:
        #     dist_0 = dist_0.unsqueeze(0)
        #     dist_1 = dist_1.unsqueeze(0)
        # dist_0 = dist_0.unsqueeze(1)
        # dist_1 = dist_1.unsqueeze(1)
        # target = target.unsqueeze(1)
        # d0s.append(dist_0.detach())
        # d1s.append(dist_1.detach())
        # targets.append(target.detach())
        # calculate_twoafc_score(d0s, d1s, targets)
        show_images(adv_img_ref)
        print(model(adv_img_ref, img_ref))
        break

    # twoafc_score = calculate_twoafc_score(d0s, d1s, targets)
    # logging.info(f"Final 2AFC score: {str(twoafc_score)}")
    # return twoafc_score


def show_images(img_ref):
    for idx, img in enumerate(img_ref):
        plt.imshow(img.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'{idx}.pdf', format="pdf", bbox_inches='tight', pad_inches=0)


def get_baseline_model(baseline_model, feat_type: str = "cls", stride: str = "16",
                       load_dir: str = "./models", device: str = "cuda"):
    if baseline_model == 'psnr':
        def psnr_func(im1, im2):
            return -peak_signal_noise_ratio(im1, im2, data_range=1.0, dim=(1, 2, 3), reduction='none')

        return psnr_func

    elif baseline_model == 'ssim':
        def ssim_func(im1, im2):
            return -structural_similarity_index_measure(im1, im2, data_range=1.0, reduction='none')

        return ssim_func

    elif baseline_model == 'dists':
        dists_metric = DISTS().to(device)

        def dists_func(im1, im2):
            distances = dists_metric(im1, im2)
            return distances

        return dists_func

    elif baseline_model == 'lpips':
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').eval()

        def lpips_func(im1, im2):
            distances = lpips_fn(im1.to(device), im2.to(device)).reshape(-1)
            return distances

        return lpips_func

    elif 'clip' in baseline_model or 'dino' in baseline_model or "open_clip" in baseline_model or "mae" in baseline_model:
        perceptual_model, _ = dreamsim(pretrained=True, dreamsim_type=args.baseline_model)
        # PerceptualModel(feat_type=feat_type, model_type=baseline_model, stride=stride,
        #                                 baseline=True, load_dir=load_dir, device=device)
        for extractor in perceptual_model.extractor_list:
            extractor.model.eval()
        return perceptual_model

    elif baseline_model == "dreamsim":
        dreamsim_model, preprocess = dreamsim(pretrained=True, cache_dir=load_dir, device=device)
        return dreamsim_model

    else:
        raise ValueError(f"Model {baseline_model} not supported.")


def run(args, device):
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.checkpoint_epoch is not None:
        if args.baseline_model is not None:
            raise ValueError("Cannot run baseline evaluation with a checkpoint.")
        args_path = os.path.join(args.eval_root, "config.yaml")
        logging.basicConfig(filename=os.path.join(args.eval_root, 'eval.log'), level=logging.INFO, force=True)
        with open(args_path) as f:
            logging.info(f"Loading checkpoint arguments from {args_path}")
            eval_args = yaml.load(f, Loader=yaml.Loader)

        eval_args.load_dir = args.load_dir
        model = LightningPerceptualModel(**vars(eval_args), device=device)
        logging.info(f"Loading checkpoint from {args.eval_root} using epoch {args.checkpoint_epoch}")

        checkpoint_root = os.path.join(args.eval_root, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_root, f"epoch={(args.checkpoint_epoch):02d}.ckpt")
        sd = torch.load(checkpoint_path)
        model.load_state_dict(sd["state_dict"])
        if eval_args.use_lora:
            model.load_lora_weights(checkpoint_root=checkpoint_root, epoch_load=args.checkpoint_epoch)
        model = model.perceptual_model
        for extractor in model.extractor_list:
            extractor.model.eval()
        model = model.to(device)
        output_path = checkpoint_root
        model_type = eval_args.model_type

    elif args.baseline_model is not None:
        if not os.path.exists(args.baseline_output_path):
            os.mkdir(args.baseline_output_path)
        logging.basicConfig(filename=os.path.join(args.baseline_output_path, 'eval.log'), level=logging.INFO,
                            force=True)
        model = get_baseline_model(args.baseline_model, args.baseline_feat_type, args.baseline_stride, args.load_dir,
                                   device)
        output_path = args.baseline_output_path
        model_type = args.baseline_model

    else:
        raise ValueError("Must specify one of checkpoint_path or baseline_model")

    eval_results = {}

    test_dataset_imagenet = TwoAFCDataset(root_dir=args.nights_root, split="test",
                                          preprocess=get_preprocess(model_type))
    # test_dataset_no_imagenet = TwoAFCDataset(root_dir=args.nights_root, split="test_no_imagenet",
    #                                          preprocess=get_preprocess(model_type))
    # total_length = len(test_dataset_no_imagenet) + len(test_dataset_imagenet)
    test_imagenet_loader = DataLoader(test_dataset_imagenet, batch_size=args.batch_size,
                                      num_workers=args.num_workers, shuffle=False)
    # test_no_imagenet_loader = DataLoader(test_dataset_no_imagenet, batch_size=args.batch_size,
    #                                      num_workers=args.num_workers, shuffle=False)

    imagenet_score = score_nights_dataset(model, test_imagenet_loader, device, args.attack_type, float(args.eps))
    # no_imagenet_score = score_nights_dataset(model, test_no_imagenet_loader, device, args.attack_type, float(args.eps))

    # eval_results['nights_imagenet'] = imagenet_score.item()
    # logging.info(f"Imagenet 2AFC score: {str(eval_results['nights_imagenet'])}")
    # eval_results['nights_no_imagenet'] = no_imagenet_score.item()
    # logging.info(f"No Imagenet 2AFC score: {str(eval_results['nights_no_imagenet'])}")
    # eval_results['nights_total'] = (imagenet_score.item() * len(test_dataset_imagenet) +
    #                                 no_imagenet_score.item() * len(test_dataset_no_imagenet)) / total_length
    # logging.info(f"Combined 2AFC score: {str(eval_results['nights_total'])}")
    #
    # logging.info(f"Saving to {os.path.join(output_path, 'eval_results.pkl')}")
    # with open(os.path.join(output_path, 'eval_results.pkl'), "wb") as f:
    #     pickle.dump(eval_results, f)
    #
    # print("Done :)")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args, device)
