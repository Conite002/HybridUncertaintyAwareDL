import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



def create_finetune_model(base_model, num_classes, trainable_layers=20):
    """Creates a fine-tuned model by unfreezing the top trainable_layers."""
    base = base_model(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    for layer in base.layers[-trainable_layers:]:
        layer.trainable = True

    x = Flatten()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    
    return model

def train_finetune_model(model, train_data, val_data, model_name, use_val=True, patience=5, epochs=20):
    """Fine-tunes the model with optional validation."""
    
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss" if use_val else "loss", factor=0.5, patience=patience, min_lr=1e-7),
        EarlyStopping(monitor="val_loss" if use_val else "loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(f"../Outputs/models/{model_name}_finetuned.h5", save_best_only=True)
    ]
    
    print(f"\n Fine-tuning {model_name} {'with' if use_val else 'without'} validation...")

    history = model.fit(
        train_data, 
        validation_data=val_data if use_val else None,
        epochs=epochs, 
        callbacks=callbacks
    )
    
    return history
