import sys
import argparse
from pixelpal.ai import train
from pixelpal.visualisation.display import display_file_or_dir
from pixelpal.evaluate import evaluate_augmentator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_parser(parser):
    parser.add_argument('module', type=str, help="Module to train")
    parser.add_argument('dataset', type=str, help="Dataset to train on")
    parser.add_argument('weights', type=str, help="Where to store the weights")
    parser.add_argument('--validation-dataset', type=str, help="Dataset to use as validation")
    parser.add_argument('--epochs', type=int, help="Number of epochs")
    parser.add_argument('--callbacks', type=str, nargs='+', help="List of callbacks")
    parser.add_argument('--data-augmentation', type=str, nargs='+', help='Data augmentation generator')


def display_parser(parser):
    parser.add_argument('image', type=str, help="Image or folder of images to displays")
    parser.add_argument('--module', type=str, help="Module to use")
    parser.add_argument('--weights', type=str, help="Weights to load")
    parser.add_argument('--horizontal-flip', type=str2bool, help="Flip image horizontally", nargs='?', const=True, default=False)
    parser.add_argument('--vertical-flip', type=str2bool, help="Flip image vertically", nargs='?', const=True, default=False)
    parser.add_argument('--output', type=str, help="Output file or folder to save to")


def evaluate_parser(parser):
    parser.add_argument('dataset', type=str, help="Image dataset")
    parser.add_argument('module', type=str, help="Module to load")
    parser.add_argument('--weights', type=str, help="Weights to load")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pixelpal")
    subparsers = parser.add_subparsers(dest='tool')
    parser_train = subparsers.add_parser('train', help='Train a model')
    train_parser(parser_train)
    parser_display = subparsers.add_parser('augment', help='Display a file')
    display_parser(parser_display)
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate quality of module')
    evaluate_parser(parser_evaluate)
    args = parser.parse_args()
    if args.tool == 'train':
        train(
            args.module, args.dataset, args.weights, 
            validation_dataset= args.validation_dataset, epochs=args.epochs, 
            callbacks=args.callbacks, data_augmentation=args.data_augmentation
        )
    elif args.tool == 'augment':
        display_file_or_dir(args.image, args.module, args.weights, horizontal_flip=args.horizontal_flip, vertical_flip=args.vertical_flip, save_file_or_dir=args.output)
    elif args.tool == 'evaluate':
        evaluate_augmentator(args.dataset + '/32x32', args.dataset + '/64x64', args.module, weights=args.weights)
    else:
        raise Exception("Requires argument, either 'train' or 'augment' or 'evaluate'")
