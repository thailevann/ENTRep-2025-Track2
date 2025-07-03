"""
Utils package for ENTRep 2025 Track 2
Contains all utility functions organized by functionality
"""

# Import from common utilities
from .common import (
    setup_device_and_clip,
    set_random_seeds,
    print_class_distribution,
    load_image_safely,
    normalize_embeddings,
    cosine_similarity_batch,
    save_json,
    load_json,
    create_directory,
    get_file_list,
    validate_embeddings,
    calculate_memory_usage,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)

# Import from data processing utilities
from .data_processing import (
    load_images_recursively,
    create_cls_map,
    generate_parent_child,
    get_hard_negative,
    create_triplets,
    filter_parent_child,
    split_embeddings,
    filter_cls_map,
    get_triplet_batch,
    calculate_embedding_stats,
    visualize_triplets
)

# Import from prediction utilities
from .prediction_utils import (
    calculate_similarity_matrix,
    get_top_k_similar,
    create_submission_format,
    compute_embedding_statistics,
    filter_predictions_by_confidence,
    compute_recall_at_k,
    compute_recall_at_1,
    compute_accuracy_at_k,
    compute_accuracy_at_1,
    evaluate_retrieval_recall,
    evaluate_retrieval_accuracy,
    compute_classification_accuracy_from_embeddings
)

# Import from training utilities
from .training_utils import (
    euler_integration,
    compute_triplet_loss,
    get_embeddings_labels_from_triplets,
    setup_optimizer_and_scheduler,
    update_learning_rate,
    check_early_stopping,
    save_checkpoint,
    load_checkpoint,
    log_training_progress,
    compute_classification_accuracy,
    compute_triplet_accuracy,
    compute_embedding_classification_accuracy
)

__all__ = [
    # Common utilities
    'setup_device_and_clip',
    'set_random_seeds',
    'print_class_distribution',
    'load_image_safely',
    'normalize_embeddings',
    'cosine_similarity_batch',
    'save_json',
    'load_json',
    'create_directory',
    'get_file_list',
    'validate_embeddings',
    'calculate_memory_usage',
    'CLASS_NAMES',
    'CLASS_TO_IDX',
    'IDX_TO_CLASS',
    
    # Data processing utilities
    'load_images_recursively',
    'create_cls_map',
    'generate_parent_child',
    'get_hard_negative',
    'create_triplets',
    'filter_parent_child',
    'split_embeddings',
    'filter_cls_map',
    'get_triplet_batch',
    'calculate_embedding_stats',
    'visualize_triplets',
    
    # Prediction utilities
    'calculate_similarity_matrix',
    'get_top_k_similar',
    'create_submission_format',
    'compute_embedding_statistics',
    'filter_predictions_by_confidence',
    'compute_recall_at_k',
    'compute_recall_at_1',
    'compute_accuracy_at_k',
    'compute_accuracy_at_1',
    'evaluate_retrieval_recall',
    'evaluate_retrieval_accuracy',
    'compute_classification_accuracy_from_embeddings',
    
    # Training utilities
    'euler_integration',
    'compute_triplet_loss',
    'get_embeddings_labels_from_triplets',
    'setup_optimizer_and_scheduler',
    'update_learning_rate',
    'check_early_stopping',
    'save_checkpoint',
    'load_checkpoint',
    'log_training_progress',
    'compute_classification_accuracy',
    'compute_triplet_accuracy',
    'compute_embedding_classification_accuracy'
]