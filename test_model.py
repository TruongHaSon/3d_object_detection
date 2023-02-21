def parse_args():
    parser = ArgumentParser()
    # Data options
    parser.add_argument('-f')
    parser.add_argument('--root', type=str, default='/content/drive/MyDrive/3d_object_detection/KITTI',
                        help='root directory of the KITTI dataset')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='gpu to use for inference (-1 for cpu)')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    parser.add_argument('--nms-thresh', type=float, default=0.2,
                        help='minimum score for a positive detection')

    # Model options
    parser.add_argument('--grid-height', type=float, default=4.,
                        help='size of grid cells, in meters')
    parser.add_argument('-r', '--grid-res', type=float, default=0.5,
                        help='size of grid cells, in meters')
    parser.add_argument('--frontend', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='name of frontend ResNet architecture')
    parser.add_argument('--topdown', type=int, default=8,
                        help='number of residual blocks in topdown network')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Load validation dataset to visualise
dataset = KittiObjectDataset(
    args.root, 'train', args.grid_size, args.grid_res, args.yoffset)
    
# Build model
model = OftNet(num_classes=1, frontend=args.frontend, 
                   topdown_layers=args.topdown, grid_res=args.grid_res, 
                   grid_height=args.grid_height)
model.cuda()
    
model_path = "/content/drive/MyDrive/3d_object_detection/backup/checkpoint-0126.pth.gz"
# Load checkpoint
ckpt = torch.load(model_path)
last_epoch = ckpt['epoch']
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optim'])
args.lr = ckpt['scheduler']['_last_lr']

# Create encoder
encoder = ObjectEncoder(nms_thresh=args.nms_thresh)

def validate(args, dataloader, model, encoder):
    
    print('\n==> Validating on {} minibatches\n'.format(len(dataloader)))
    model.eval()
    epoch_loss = MetricDict()

    for i, (_, image, calib, objects, grid) in enumerate(dataloader):


        image, calib, grid = image.cuda(), calib.cuda(), grid.cuda()

        with torch.no_grad():

            # Run network forwards
            pred_encoded = model(image, calib, grid)

            # Encode ground truth objects
            gt_encoded = encoder.encode_batch(objects, grid)
        
            # Decode predictions
            preds = encoder.decode_batch(*pred_encoded, grid)

            # Visualize scores
            visualize_score(pred_encoded[0], gt_encoded[0], grid)
            plt.show()
            visualise_bboxes(image, calib, objects, preds)
            plt.show()


a = iter(dataset)
idx, image, calib, objects, grid = next(a)
validate(args, val_loader, model, encoder)
