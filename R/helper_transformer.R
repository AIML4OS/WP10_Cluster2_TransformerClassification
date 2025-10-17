# function to build the transformer
transformer_encoder <- function(inputs,
                                embed_dim,
                                num_heads,
                                ff_dim,
                                dropout = 0) {
  # Attention and Normalization
  attention_layer <-
    layer_multi_head_attention(key_dim = embed_dim,
                               num_heads = num_heads,
                               dropout = dropout)
  
  x <- inputs %>% 
    attention_layer(., .) %>% 
    layer_dropout(dropout)
  
  res <- x + inputs   %>% 
    layer_layer_normalization(epsilon = 1e-3)
  
  # Feed Forward Network
  x <- res |>
    layer_conv_1d(ff_dim, kernel_size = 1, activation = "relu") |>
    layer_conv_1d(embed_dim, kernel_size = 1) |> 
    layer_dropout(dropout)
  
  x <- x + res |>
    layer_layer_normalization(epsilon = 1e-3)
  
  return(x)
}

build_model <- function(inputs,
                        embed_dim,
                        num_heads,
                        ff_dim,
                        num_transformer_blocks,
                        maxlen,
                        dense_dim,
                        num_words,
                        n_classes,
                        regularization,
                        dropout1 = 0,
                        dropout2 = 0) {
  
  x <- inputs |> layer_embedding(num_words,
                                 output_dim = embed_dim)
  
  positions  <- tf$range(start=0, limit = maxlen, delta = 1)
  positions <- positions |> layer_embedding(input_dim = maxlen, 
                                            output_dim = embed_dim)
  
  x <- x + positions
  
  x <- x |>
    layer_layer_normalization()
  
  for (i in 1:num_transformer_blocks) {
    x <- x |>
      transformer_encoder(
        embed_dim = embed_dim,
        num_heads = num_heads,
        ff_dim = ff_dim,
        dropout = dropout1
      )
  }
  
  x <- x |> 
    layer_global_average_pooling_1d(data_format = "channels_first")
  
  x <- x |>
    layer_dense(dense_dim,activation = "relu") |>
    layer_dropout(dropout2)
  
  return(x)
  # print("dense layer done")
  # outputs <- x |> 
  #   layer_batch_normalization() |>
  #   layer_dense(n_classes, activation = "softmax",
  #               kernel_regularizer = regularizer_l2(l=regularization))
  # print("dense layer2 done")
  # 
  # print(paste("input shape:",inputs$shape))
  # print(paste("output shape:",outputs$shape))
  # 
  # keras_model(inputs, outputs)
}

build_layer <- function(input_shape,
                        embed_dim,
                        num_heads,
                        ff_dim,
                        num_transformer_blocks,
                        maxlen,
                        dense_dim,
                        num_words,
                        n_classes,
                        regularization,
                        dropout1 = 0,
                        dropout2 = 0) {
  
  inputs <- layer_input(shape=input_shape)
  
  x <- inputs |> layer_embedding(num_words,
                                 output_dim = embed_dim
                                 #,mask_zero = TRUE
  )
  
  positions  <- tf$range(start=0, limit = maxlen, delta = 1)
  positions <- positions |> layer_embedding(input_dim = maxlen, 
                                            output_dim = embed_dim
                                            #,mask_zero = TRUE
  )
  
  x <- x + positions
  
  x <- x |>
    layer_layer_normalization()
  
  for (i in 1:num_transformer_blocks) {
    x <- x |>
      transformer_encoder(
        embed_dim = embed_dim,
        num_heads = num_heads,
        ff_dim = ff_dim,
        dropout = dropout1
      )
  }
  
  x <- x |> 
    layer_global_average_pooling_1d(data_format = "channels_first")
  
  x <- x |>
    layer_dense(dense_dim,activation = "relu") |>
    layer_dropout(dropout2)
  
  return(list(x_layer=x,input=inputs))
}

build_model_stacked <- function(x_layer1, x_layer2, input1, input2, n_classes){
  model_deep <- layer_concatenate(list(x_layer1, x_layer2))
  inputs_all <- c(input1,input2)
  
  outputs <- model_deep |> 
    layer_batch_normalization() |>
    layer_dense(n_classes, activation = "softmax",
                kernel_regularizer = regularizer_l2(FLAGS$regularization))
  
  keras_model(inputs_all, list(outputs))
}
