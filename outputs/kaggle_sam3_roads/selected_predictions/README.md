# Selected prediction examples

Ten examples are included for each evaluated split:

- `validation/`: 10 validation examples
- `test/`: 10 held-out test examples

Each sample includes four aligned files:

- `*_image.png`: RGB input image
- `*_ground_truth.png`: reference road mask
- `*_prediction.png`: predicted road mask
- `*_triplet.png`: combined image, ground truth, and prediction panel

The complete `validation_predictions/` and `test_predictions/` directories are
generated locally by the evaluation command but are not tracked in Git because
they contain hundreds of megabytes of rendered images.
