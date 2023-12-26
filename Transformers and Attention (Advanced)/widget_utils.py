import ipywidgets as widgets
from IPython.display import display

# Define global variables with default values
BATCH_SIZE = 128
T = 80
D_K = 16
D_V = 16
D_MODEL = 128
H = 8
VOCAB_SIZE = 10
N_TRANSFORMERS_BLOCKS_ENCODER = 6
N_CLASSES = 2
USE_MASK = False
MASK_BEFORE_SOFTMAX = True
LOG_INTERVAL_IN_BATCHES = 10
EPOCHS = 12
ALLOW_GPU = True
RUN_UNIT_TESTS = True
TRAIN_MODEL = False
SEED = 42


# Define update functions
def update_BATCH_SIZE(change):
    global BATCH_SIZE
    BATCH_SIZE = change.new


def update_T(change):
    global T
    T = change.new


def update_D_K(change):
    global D_K
    D_K = change.new


def update_D_V(change):
    global D_V
    D_V = change.new


def update_D_MODEL(change):
    global D_MODEL
    D_MODEL = change.new


def update_H(change):
    global H
    H = change.new


def update_VOCAB_SIZE(change):
    global VOCAB_SIZE
    VOCAB_SIZE = change.new


def update_N_TRANSFORMERS_BLOCKS_ENCODER(change):
    global N_TRANSFORMERS_BLOCKS_ENCODER
    N_TRANSFORMERS_BLOCKS_ENCODER = change.new


def update_N_CLASSES(change):
    global N_CLASSES
    N_CLASSES = change.new


def update_USE_MASK(change):
    global USE_MASK
    USE_MASK = change.new


def update_MASK_BEFORE_SOFTMAX(change):
    global MASK_BEFORE_SOFTMAX
    MASK_BEFORE_SOFTMAX = change.new


def update_LOG_INTERVAL_IN_BATCHES(change):
    global LOG_INTERVAL_IN_BATCHES
    LOG_INTERVAL_IN_BATCHES = change.new


def update_EPOCHS(change):
    global EPOCHS
    EPOCHS = change.new


def update_ALLOW_GPU(change):
    global ALLOW_GPU
    ALLOW_GPU = change.new


def update_RUN_UNIT_TESTS(change):
    global RUN_UNIT_TESTS
    RUN_UNIT_TESTS = change.new


def update_TRAIN_MODEL(change):
    global TRAIN_MODEL
    TRAIN_MODEL = change.new


def update_SEED(change):
    global SEED
    SEED = change.new


def initialize_widgets():
    global batch_size_w, t_w, d_k_w, d_v_w, d_model_w, h_w, vocab_size_w, n_transformers_blocks_encoder_w, n_classes_w, use_mask_w, mask_before_softmax_w, log_interval_in_batches_w, epochs_w, allow_gpu_w, run_unit_tests_w, train_model_w, seed_w
    # Create widgets
    batch_size_w = widgets.IntText(
        value=128, description="BATCH_SIZE:", style={"description_width": "initial"}
    )
    t_w = widgets.IntText(
        value=80, description="T:", style={"description_width": "initial"}
    )
    d_k_w = widgets.IntText(
        value=64, description="D_K:", style={"description_width": "initial"}
    )
    d_v_w = widgets.IntText(
        value=64, description="D_V:", style={"description_width": "initial"}
    )
    d_model_w = widgets.IntText(
        value=128, description="D_MODEL:", style={"description_width": "initial"}
    )
    h_w = widgets.IntText(
        value=8, description="H:", style={"description_width": "initial"}
    )
    vocab_size_w = widgets.IntText(
        value=10, description="VOCAB_SIZE:", style={"description_width": "initial"}
    )
    n_transformers_blocks_encoder_w = widgets.IntText(
        value=6,
        description="N_TRANSFORMERS_BLOCKS_ENCODER:",
        style={"description_width": "initial"},
    )
    n_classes_w = widgets.IntText(
        value=2, description="N_CLASSES:", style={"description_width": "initial"}
    )
    use_mask_w = widgets.Checkbox(
        value=False, description="USE_MASK:", style={"description_width": "initial"}
    )
    mask_before_softmax_w = widgets.Checkbox(
        value=True,
        description="MASK_BEFORE_SOFTMAX:",
        style={"description_width": "initial"},
    )
    log_interval_in_batches_w = widgets.IntText(
        value=10,
        description="LOG_INTERVAL_IN_BATCHES:",
        style={"description_width": "initial"},
    )
    epochs_w = widgets.IntText(
        value=12, description="EPOCHS:", style={"description_width": "initial"}
    )
    allow_gpu_w = widgets.Checkbox(
        value=True, description="ALLOW_GPU:", style={"description_width": "initial"}
    )
    run_unit_tests_w = widgets.Checkbox(
        value=True,
        description="RUN_UNIT_TESTS:",
        style={"description_width": "initial"},
    )
    train_model_w = widgets.Checkbox(
        value=False, description="TRAIN_MODEL:", style={"description_width": "initial"}
    )
    seed_w = widgets.IntText(
        value=42, description="SEED:", style={"description_width": "initial"}
    )

    # Attach the update functions to the widgets
    batch_size_w.observe(update_BATCH_SIZE, "value")
    t_w.observe(update_T, "value")
    d_k_w.observe(update_D_K, "value")
    d_v_w.observe(update_D_V, "value")
    d_model_w.observe(update_D_MODEL, "value")
    h_w.observe(update_H, "value")
    vocab_size_w.observe(update_VOCAB_SIZE, "value")
    n_transformers_blocks_encoder_w.observe(
        update_N_TRANSFORMERS_BLOCKS_ENCODER, "value"
    )
    n_classes_w.observe(update_N_CLASSES, "value")
    use_mask_w.observe(update_USE_MASK, "value")
    mask_before_softmax_w.observe(update_MASK_BEFORE_SOFTMAX, "value")
    log_interval_in_batches_w.observe(update_LOG_INTERVAL_IN_BATCHES, "value")
    epochs_w.observe(update_EPOCHS, "value")
    allow_gpu_w.observe(update_ALLOW_GPU, "value")
    run_unit_tests_w.observe(update_RUN_UNIT_TESTS, "value")
    train_model_w.observe(update_TRAIN_MODEL, "value")
    seed_w.observe(update_SEED, "value")

    # Create grid layouts
    grid1 = widgets.GridBox(
        [
            batch_size_w,
            t_w,
            d_k_w,
            d_v_w,
            d_model_w,
            h_w,
            vocab_size_w,
            n_transformers_blocks_encoder_w,
        ],
        layout=widgets.Layout(
            grid_template_columns="repeat(3, 300px)",
            grid_template_rows="repeat(2, 50px)",
            grid_gap="10px",
        ),
    )
    grid2 = widgets.GridBox(
        [n_classes_w],
        layout=widgets.Layout(
            grid_template_columns="repeat(1, 300px)", grid_gap="10px"
        ),
    )
    grid3 = widgets.GridBox(
        [use_mask_w, mask_before_softmax_w],
        layout=widgets.Layout(
            grid_template_columns="repeat(2, 300px)", grid_gap="10px"
        ),
    )
    grid4 = widgets.GridBox(
        [log_interval_in_batches_w],
        layout=widgets.Layout(
            grid_template_columns="repeat(1, 300px)", grid_gap="10px"
        ),
    )
    grid5 = widgets.GridBox(
        [epochs_w, run_unit_tests_w, train_model_w],
    )
    grid6 = widgets.GridBox(
        [allow_gpu_w],
        layout=widgets.Layout(
            grid_template_columns="repeat(1, 300px)", grid_gap="10px"
        ),
    )
    grid7 = widgets.GridBox(
        [seed_w],
        layout=widgets.Layout(
            grid_template_columns="repeat(1, 300px)", grid_gap="10px"
        ),
    )

    # Create labels
    label1 = widgets.Label(value="Hyperparameters:")
    label2 = widgets.Label(value="Classes:")
    label3 = widgets.Label(value="Mask Settings:")
    label4 = widgets.Label(value="Logging Interval:")
    label5 = widgets.Label(value="Epochs and Settings:")
    label6 = widgets.Label(value="GPU Settings:")
    label7 = widgets.Label(value="Seed:")

    # Display the labels and grids
    display(label1)
    display(grid1)
    print("\n")
    display(label2)
    display(grid2)
    print("\n")
    display(label3)
    display(grid3)
    print("\n")
    display(label4)
    display(grid4)
    print("\n")
    display(label5)
    display(grid5)
    print("\n")
    display(label6)
    display(grid6)
    print("\n")
    display(label7)
    display(grid7)
