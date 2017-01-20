# from faster_rcnn.huey_queue_config import huey

from faster_rcnn.sift_svm import (
    SiftFeaturesPreparator,
    train_svm, create_dataset_for_svm
)


# @huey.task()
def train_exemplar_svm_on_sift_features(
    original_image,
    cropped_positive_image,
    cropped_image_point_on_original,
    dense_sift=True, clustering='kmeans',
    augment_data=True):
    X, y = create_dataset_for_svm(
        original_image, cropped_positive_image,
        cropped_image_point_on_original, augment_data
    )

    X = SiftFeaturesPreparator(dense_sift=dense_sift, clustering=clustering).fit_transform(X)
    model = train_svm(X, y)
    return model
