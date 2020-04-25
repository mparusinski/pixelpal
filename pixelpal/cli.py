import argparse
from pixelpal.ai import train
from pixelpal.visualisation.display import display_file


def train_parser(parser):
    parser.add_argument('module', type=str, help="Module to train")
    parser.add_argument('dataset', type=str, help="Dataset to train on")
    parser.add_argument('weights', type=str, help="Where to store the weights")
    parser.add_argument('--validation-dataset', type=str, help="Dataset to use as validation")
    parser.add_argument('--epochs', type=int, help="Number of epochs")
    parser.add_argument('--callbacks', type=str, nargs='+', help="List of callbacks")


def display_parser(parser):
    parser.add_argument('image', type=str, help="Image to display")
    parser.add_argument('--module', type=str, help="Module to use")
    parser.add_argument('--weights', type=str, help="Weights to load")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pixelpal")
    subparsers = parser.add_subparsers(dest='tool')
    
    parser_train = subparsers.add_parser('train', help='Train a model')
    train_parser(parser_train)
    
    parser_display = subparsers.add_parser('display', help='Display a file')
    display_parser(parser_display)
    
    args = parser.parse_args()
    if args.tool == 'train':
        train(args.module, args.dataset, args.weights, validation_dataset= args.validation_dataset, epochs=args.epochs, callbacks=args.callbacks)
        
    if args.tool == 'display':
        display_file(args.image, args.module, args.weights)
