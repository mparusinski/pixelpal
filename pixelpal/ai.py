import os

from pixelpal.model.base import learn_with_gan, learn, get_model, save_weights, get_callbacks, get_data_augmentation


def train(module, dataset, weights, **kwargs):
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    model = get_model(module)
    callbacks = get_callbacks(kwargs.get('callbacks', []))
    data_augmentation = get_data_augmentation(kwargs.get('data_augmentation', []))
    use_gan_loss = kwargs.get('use_gan_loss', False)

    if use_gan_loss:
        learn_with_gan(
            model, dataset, validation_dataset=kwargs.get('validation_dataset'), callbacks=callbacks,
            data_augmentation=data_augmentation
        )
    else:
        learn(
            model, dataset, validation_dataset=kwargs.get('validation_dataset'), callbacks=callbacks, 
            data_augmentation=data_augmentation
        )
    save_weights(model, weights)


