# A list of Keras Application image models with their approximate parameter counts and disk size.
# Note: Parameter counts are for the full model including the top classification layer (in ours classification layer is not used, so will be less).
# Size is calculated assuming FP32 precision (4 bytes per parameter).

image_model_ids = [
    "DenseNet121",  # ~8 Million params, ~31 MB
    "DenseNet169",  # ~14 Million params, ~54 MB
    "DenseNet201",  # ~20 Million params, ~77 MB
    "EfficientNetB0",  # ~5.3 Million params, ~21 MB
    "EfficientNetB1",  # ~7.8 Million params, ~30 MB
    "EfficientNetB2",  # ~9.2 Million params, ~35 MB
    "EfficientNetB3",  # ~12 Million params, ~46 MB
    "EfficientNetB4",  # ~19 Million params, ~73 MB
    "EfficientNetB5",  # ~30 Million params, ~115 MB
    "EfficientNetB6",  # ~43 Million params, ~164 MB
    "EfficientNetB7",  # ~66 Million params, ~252 MB
    "EfficientNetV2B0",  # ~7.1 Million params, ~27 MB
    "EfficientNetV2B1",  # ~8.2 Million params, ~32 MB
    "EfficientNetV2B2",  # ~10.1 Million params, ~39 MB
    "EfficientNetV2B3",  # ~14.4 Million params, ~55 MB
    "EfficientNetV2L",  # ~119.5 Million params, ~456 MB
    "EfficientNetV2M",  # ~54.1 Million params, ~207 MB
    "EfficientNetV2S",  # ~21.5 Million params, ~82 MB
    "InceptionResNetV2",  # ~55.8 Million params, ~213 MB
    "InceptionV3",  # ~23.8 Million params, ~91 MB
    "MobileNet",  # ~4.2 Million params, ~16 MB
    "MobileNetV2",  # ~3.5 Million params, ~14 MB
    "MobileNetV3Large",  # ~5.5 Million params, ~21 MB
    "MobileNetV3Small",  # ~2.5 Million params, ~10 MB
    "NASNetLarge",  # ~88.9 Million params, ~339 MB
    "NASNetMobile",  # ~5.3 Million params, ~21 MB
    "ResNet101",  # ~44.5 Million params, ~170 MB
    "ResNet101V2",  # ~44.6 Million params, ~170 MB
    "ResNet152",  # ~60.2 Million params, ~230 MB
    "ResNet152V2",  # ~60.4 Million params, ~231 MB
    "ResNet50",  # ~25.6 Million params, ~98 MB
    "ResNet50V2",  # ~25.6 Million params, ~98 MB
    "VGG16",  # ~138 Million params, ~527 MB
    "VGG19",  # ~143 Million params, ~546 MB
    "Xception",  # ~22.9 Million params, ~88 MB
]

# Note: Size is calculated assuming FP32 precision (4 bytes per parameter).
text_model_ids = [
    "bert-base-uncased",  # ~110 Million params, ~420 MB
    "distilbert-base-uncased",  # ~66 Million params, ~252 MB
    "sentence-transformers/all-mpnet-base-v2",  # ~110 Million params, ~420 MB
    "sentence-transformers/all-MiniLM-L6-v2",  # ~23 Million params, ~88 MB
    "microsoft/MiniLM-L12-H384-uncased",  # ~33 Million params, ~126 MB
    "huawei-noah/TinyBERT_General_4L_312D",  # ~15 Million params, ~58 MB
    "google/bert_uncased_L-4_H-256_A-4",  # ~12 Million params, ~46 MB
    "prajjwal1/bert-tiny",  # ~4.4 Million params, ~17 MB
    "roberta-base",  # ~125 Million params, ~477 MB
    "facebook/bart-base",  # ~140 Million params, ~534 MB
    "bert-large-uncased",  # ~335 Million params, ~1.3 GB
    "google/electra-small-discriminator",  # ~14 Million params, ~54 MB
]
