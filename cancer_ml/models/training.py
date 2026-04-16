"""
Helper functions for general model training.
"""
import keras
import wandb
from wandb.integration.keras import WandbMetricsLogger

def fit_and_evaluate(
        model: keras.Model,
        dsets: dict,
        paths: dict,
        config: dict,
        callbacks: list | None = None,
        verbose: int = 1,
) -> None:
    """
    Fit and evaluate a model.
    """
    # set paths
    model_dir = paths["model"]
    best_weights_file = model_dir / "best.weights.h5"
    final_weights_file = model_dir / "final.weights.h5"
    csv_file = model_dir / "log.csv"

    # inuit wandb
    wandb.init(
        project="cancer_ml",
        config=config,
        name=config["meta"]["model_id"],
        dir=paths["wandb"].parent,
    )

    # prepare some standard callbacks
    if callbacks is None:
        callbacks = []
    extra_callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_weights_file,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_dice"),
        keras.callbacks.CSVLogger(csv_file),
        WandbMetricsLogger(),
    ]
    callbacks.extend(extra_callbacks)

    print("---Fit---")
    model.fit(
        dsets["train"],
        validation_data=dsets["val"],
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=verbose,
    )
    model.save_weights(final_weights_file)

    print("---Evaluate---")
    model.load_weights(best_weights_file)
    scores = model.evaluate(
        dsets["test"],
        verbose=verbose,
    )
    scores = {
        "test_loss": scores[0],
        "test_dice": scores[1],
    }
    wandb.log(scores)
    wandb.finish()