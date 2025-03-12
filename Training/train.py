import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def train_model(model, train_data, val_data, model_name, use_val=True):
    """Trains the model with optional validation."""
    
    callbacks = [
        EarlyStopping(monitor="val_loss" if use_val else "loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(f"../Outputs/models/{model_name}.h5", save_best_only=True)
    ]
    
    print(f"\nTraining {model_name} {'with' if use_val else 'without'} validation...")
    
    history = model.fit(
        train_data, 
        validation_data=val_data if use_val else None,
        epochs=25, 
        callbacks=callbacks
    )
    
    return history


