local is_cose = if std.findSubstr('cose', std.extVar('TRAIN_DATA_PATH')) == [] then false else true;

{
  dataset_reader : {
    type : "rationale_reader_beer_nosep",
    token_indexers : {
      bert : {
        type : "bert-pretrained",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
  },
  validation_dataset_reader: {
    type : "rationale_reader_beer_nosep",
    token_indexers : {
      bert : {
        type : "bert-pretrained",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "encoder_generator_rationale_model",
    generator: {
      type: "simple_generator_model",
      text_field_embedder: {
        allow_unmatched_keys: true,
        embedder_to_indexer_map: {
          bert: ["bert", "bert-offsets"],
        },
        token_embedders: {
          bert: {
            type: "bert-pretrained",
            pretrained_model: 'bert-base-uncased',
            requires_grad: false,
          },
        },
      },
      seq2seq_encoder : {
        type: 'lstm',
        input_size: 768,
        hidden_size: 128,
        num_layers: 1,
        bidirectional: true
      },
      dropout: 0.2,
      feedforward_encoder:{
        type: 'pass_through',
        input_dim: 256
      },
    },
    encoder : {
      type: "encoder_rationale_model" + (if is_cose then '_cose' else ''),
      text_field_embedder: {
        allow_unmatched_keys: true,
        embedder_to_indexer_map: {
          bert: ["bert", "bert-offsets"],
        },
        token_embedders: {
          bert: {
            type: "bert-pretrained",
            pretrained_model: 'bert-base-uncased',
            requires_grad: false,
          },
        },
      },
      seq2seq_encoder : {
        type: 'lstm',
        input_size: 768,
        hidden_size: 128,
        num_layers: 1,
        bidirectional: true
      },
      dropout: 0.2,
      attention: {
        type: 'additive',
        vector_dim: 256,
        matrix_dim: 256,
      },
      feedforward_encoder:{
        input_dim: 256,
        num_layers: 1,
        hidden_dims: [128],
        activations: ['relu'],
        dropout: 0.2
      },
    },
    reg_loss_lambda: 25.0,
    reg_loss_mu: 2.0,
    reinforce_loss_weight: 1.0,
    rationale_supervision_loss_weight: std.extVar('rs_weight')
  },
  iterator: {
    type: "bucket",
    sorting_keys: [['document', 'num_tokens']],
    batch_size : std.extVar("batch_size"),
    biggest_batch_first: true
  },
  trainer: {
    num_epochs: 1,
    patience: 15,
    grad_norm: 10.0,
    validation_metric: "+accuracy",
    num_serialized_models_to_keep: 1,
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adam",
      lr: 2e-5
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
