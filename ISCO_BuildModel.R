#################################################
# ISCO Classification NNet Transformer + One Hot
#

library(data.table)
library(keras)
library(tensorflow)
library(progress)

source("R/helpers.R")
source("R/helper_transformer.R")
params <- fread("data/best_params.csv")

# some parameter settings for tokenization and training
roll <- FALSE
keep_spaces <- TRUE
oneHot <- TRUE #whether to use oneHot encoded tokens
useWeights <- TRUE #use weights in the model

# path to save all outputs to
localPath <- as.character(Sys.Date())

if(!dir.exists(localPath)){
  dir.create(localPath)
}

# Hyperparameter flags ---------------------------------------------------
# FLAGS are used for Parametertuning: https://tensorflow.rstudio.com/guides/tfruns/tuning
FLAGS <- flags(
  # transformer flags
  flag_numeric("dropout1", params$dropout1),
  flag_numeric("dropout2", params$dropout2),
  flag_numeric("num_heads", params$num_heads), #number of transformer heads
  flag_numeric("num_transformer_blocks",params$num_trans_blocks), #number of transformer block
  flag_numeric("ff_dim",params$hidden_layer_dim),
  flag_numeric("maxlen",50),
  flag_numeric("embedding_dims",params$embed_dim),
  flag_numeric("dense_dim",params$dense_dim),
  flag_numeric("nGram1",3),
  flag_numeric("nGram2",5),
  
  
  # one hot layer
  flag_numeric("dropoutOnehot",0.3),
  flag_integer("unitsOnehot",250),
  
  # second to last relu layer
  flag_numeric("dropoutrelu",.1), # dropout for checkbox layer 
  flag_integer("unitsrelu",150), # untis after checkbox layer
  
  # model training
  flag_integer("batch_size",1000), # number of records processed in one go
  flag_integer("epochs",200), # maximum number of epochs
  flag_numeric("regular",0.001), # regularization
  flag_numeric("learningrate", 0.001),
  flag_numeric("min_delta",0.003),
  
  flag_boolean("roll", roll),
  flag_boolean("keep_spaces", keep_spaces),
  
  flag_string("Target","Code08")
)


# define params for applying str distance + model
params <- list(Index = "ID",
               method = c("osa","qgram"),
               q = 2,
               probTH = 0.91)

params$NeuralNetworkParams <- FLAGS

# set XGBoost hyperparameter (from the XGBoost library)
params$XGBoost$max.depth <- 16
params$XGBoost$eta <- 0.05
params$XGBoost$subsample <- 0.7
params$XGBoost$colsample_bytree <- 0.7

params$date <- Sys.Date()

# ------------------------------------------------------------------------------------------------------
# load data
source("R/load_data.R")

# set additional variables as either factor, character (categories) or integer vars 
# if none, set factor_vars <- NULL
factor_vars <- c("edu","citizen")

dat[,c(FLAGS$Target):=factor(get(FLAGS$Target))]


# prep data
dat <- dat[!is.na(Code08)]
dat[,Text_clean:=tolower(Text)]
# replace special characters
umlaut <- c("ä","ö","ü","ä","ö","ü","&","ß","ä","ü","ß","ö","","ü","ß",
            "ä","ä","","ö","ü","ä","ü")
kodierung <- c("ã„","ã–","ãœ","ã¤","ã¶","ã¼","&amp;","ãÿ","„","ç¬","ãÿ",
               "ã", 'â“',"š","á","ž","çï","&amp","”","Ã¼","Ã¤","\u0081")
dat[,Text_clean:=stri_replace_all_regex(Text_clean, kodierung, umlaut, vectorize_all = FALSE)]

params$umlaut <- umlaut
params$kodierung <- kodierung

# the simplify_text() function removes gender-specific word endings, special characters etc.
# (rigth now this is tailored to the german langugage and would have to be adapted)
dat[,Text_clean:=simplify_text(Text_clean)]
dat <- dat[Text_clean!=""] #remove empty strings


# cut of strings after 90 Characters
dat[,Text_clean:=substr(Text_clean,1,90)] 
params$MaxLength <- 90


# factors to integer values
if(!is.null(factor_vars)){
  factor_newnames <- paste0(factor_vars, "_INT")
  dat[,c(factor_newnames):=lapply(.SD, factor), .SDcols=c(factor_vars)]
}




# ------------------------------------------------------------------------------------------------------
# Setup INPUTS
# setup input for FACTORS
if(!is.null(factor_vars)){
  x.factors.help <- dat[,.SD,.SDcols=c(factor_vars)]
  x.factors.help <- lapply(x.factors.help, factor)
  params$help_factor_lookup <- lapply(x.factors.help,levels) #save to decode later
  x.factors.help <- lapply(x.factors.help,as.numeric)
  
  uniqueFactor <- lapply(x.factors.help,uniqueN)
  params$uniqueFactor <- uniqueFactor
  
  # one hot encoding of factor variables
  x.factors <- list()
  for(i in 1:length(uniqueFactor)){
    x.factors[[i]] <-  to_categorical(x.factors.help[[i]]-1,num_classes = uniqueFactor[[i]])
  }
  x.factors <- do.call(cbind,x.factors)

}

# tokenized texts with first nGram (here 3-gram)
x_emb_1 <- setupEmbedding2(dat,ngram=FLAGS$nGram1,string_col = "Text_clean", roll = roll, keep_spaces = keep_spaces)
x_1 <- x_emb_1[[1]] #tokenized texts with zero-padding to length of the longest token sequence
num_words_1 <- x_emb_1[[2]] #number of unique tokens for first nGram
tok_1 <- x_emb_1[[3]] #tokenizer: ngram to token ID translation

if(nrow(x_1) != nrow(dat)){
  stop("Tokenization failed")
}
saveRDS(tok_1, file = file.path(localPath,paste0("tokenizerFirstGram.RDS")))

# same as first nGram (here 5-Gram)
x_emb_2 <- setupEmbedding2(dat,ngram=FLAGS$nGram2,string_col = "Text_clean", roll = roll, keep_spaces = keep_spaces)
x_2 <- x_emb_2[[1]]
num_words_2 <- x_emb_2[[2]]
tok_2 <- x_emb_2[[3]]

if(nrow(x_2) != nrow(dat)){
  stop("Tokenization failed")
}
saveRDS(tok_2, file = file.path(localPath,paste0("tokenizerSecondGram.RDS")))

dim(x_1)
dim(x_2)

params$shape_ngram1 <- ncol(x_1)
params$shape_ngram2 <- ncol(x_2)

# target classes must be integer values from 0 to length(unique(classes))-1
dat[,ISCO_target:=as.numeric(factor(get(FLAGS$Target)))-1]
y_lookup <- unique(dat[,.(get(FLAGS$Target),ISCO_target)]) #translates ISCO Code to class integer ID

saveRDS(as.character(y_lookup[order(ISCO_target)]$V1),file=file.path(localPath,"outputLevels.RDS"))

# ------------------------------------------------------------------------------------------------------
# split in test/train/validation
set.seed(123456)

index.train <- 1:nrow(dat)
index.valid <- sample(index.train, floor(length(index.train)*0.2))

index.train <- index.train[!index.train %in% c(index.valid)]

index.test <- sample(index.train, floor(length(index.train)*0.2))
index.train <- index.train[!index.train %in% c(index.test)]



# build one hot encoding matrix from tokenized inputs
# one hot encode 
x <- setupOneHot(x_1) #first nGram  
x2 <- setupOneHot(x_2) #second nGram

params$OneHot$x1_info <- x[[2]] #data.table with TokenID - Position in onehot matrix - IDF value
params$OneHot$x2_info <- x2[[2]]

x <- x[[1]] #sparse matrices with IDF for tokens in the text - 0/. else
x2 <- x2[[1]]

gc()

# weights by occurence of ISCO Codes
case_weights <- dat[index.train,sum(count),by=c("ISCO_target")]
max_count_case <- max(case_weights$V1)
case_weights <- case_weights[,.(ISCO_target,max_count_case/V1)]
case_weights_names <- case_weights$ISCO_target
case_weights <- as.list(case_weights$V2)
names(case_weights) <- as.character(case_weights_names)


# save parameters used in script
saveRDS(params,file=file.path(localPath,"parameter.RDS"))



# define neural network architecture  ---------------------------------------------------

# Factor variable input (edu and citizenship)
if(!is.null(factor_vars)){
  input_length <- ncol(x.factors)
  input_factors <- layer_input(shape=input_length, name="FACTORS")
  model_factors <- input_factors
}


# one hot token ID input
input_length <- ncol(x) + ncol(x2)
input_onehot <- layer_input(shape=input_length, name="OneHot")
model_onehot <- input_onehot %>%
  layer_dropout(FLAGS$dropoutOnehot)%>%
  layer_dense(FLAGS$unitsOnehot,activation = "relu")%>%
  layer_batch_normalization()


# first ngram embedding followed by transformer layer
input_length <- ncol(x_1)
input_tr <- layer_input(shape=input_length, name="Transformer_3Gram")

model_tr <- build_model(
  input_tr,
  num_heads = FLAGS$num_heads,
  ff_dim = FLAGS$ff_dim, # hidden layer size in transformer layer
  num_transformer_blocks = FLAGS$num_transformer_blocks,
  dense_dim = FLAGS$dense_dim,
  dropout1 = FLAGS$dropout1,
  dropout2 = FLAGS$dropout2,
  maxlen= min(input_length,FLAGS$maxlen),
  embed_dim=FLAGS$embedding_dim,
  num_words = num_words_1
)

# second ngram embedding followed by transformer layer
input_length <- ncol(x_2)
input_tr5 <- layer_input(shape=input_length, name="Transformer_5Gram")
model_tr5 <- build_model(
  input_tr5,
  num_heads = FLAGS$num_heads,
  ff_dim = FLAGS$ff_dim, # hidden layer size in transformer layer
  num_transformer_blocks = FLAGS$num_transformer_blocks,
  dense_dim = FLAGS$dense_dim,
  dropout1 = FLAGS$dropout1,
  dropout2 = FLAGS$dropout2,
  maxlen = min(input_length,FLAGS$maxlen),
  embed_dim = FLAGS$embedding_dim,
  num_words = num_words_2
)


# concatenate all above defined layers
model_list <- list(model_tr, model_tr5) 
inputsAll <- c(input_tr,input_tr5)
if(!is.null(factor_vars)){
  model_list <- c(list(model_factors),model_list)
  inputsAll <- c(input_factors,inputsAll)
}
if(oneHot){
  model_list <- c(list(model_onehot),model_list)
  inputsAll <- c(input_onehot, inputsAll)
}
model <- layer_concatenate(model_list)

# relu layers
model <- model%>%
  layer_dropout(FLAGS$dropoutrelu)%>%
  layer_dense(FLAGS$unitsrelu,activation = "relu")

# final layer (classification layer)
model <- model%>%
  layer_batch_normalization() %>%
  layer_dense(nrow(y_lookup),activation="softmax",
              kernel_regularizer = regularizer_l2(l=FLAGS$regular))

# compile model and set compiler flags
model <- keras_model(
  inputs=inputsAll,
  outputs=list(model)
)

summary(model)

# define top 5 and top-10 accuracy functions
metric_top_5_categorical_accuracy <-
  custom_metric("top_5_categorical_accuracy", function(y_true, y_pred) {
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 5)
  })

metric_top_10_categorical_accuracy <-
  custom_metric("top_10_categorical_accuracy", function(y_true, y_pred) {
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 10)
  })

# compile the model
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  weighted_metric = list("sparse_categorical_accuracy", metric_top_10_categorical_accuracy, metric_top_5_categorical_accuracy)
)


# training model  ---------------------------------------------------
# training is done on test set, the valid set is used for optimization during training
# once the model is trained, the test set is used to evaluate the models finals oos performance 

# define some training hyperparameters
callbacks <- list(keras::callback_early_stopping(monitor = "val_loss", 
                                                 min_delta = 0.01, 
                                                 patience = 10, 
                                                 verbose = 1, 
                                                 restore_best_weights=TRUE))


######
# train on batch to minimize memory usage
batch_size <- 1000
num_steps <- ceiling(length(index.train)/batch_size)
eval_model <- c(loss = 1000, #initial model performance
                accuracy=0,
                top_5_categorical_accuracy = 0,
                top_10_categorical_accuracy = 0)
max_epochs <- FLAGS$epochs
topk <- "top_5_categorical_accuracy" #metrics to optimize for (must be in eval_model)
current_best <- eval_model[topk]
current_best_m <- 1
patience <- 10
for(m in 1:max_epochs){
  message("Epoch ",m,"/",max_epochs)
  num_steps <- ceiling(length(index.train)/batch_size)

  all_samples <- index.train
  all_samples <- sample(all_samples, size=length(all_samples))
  
  # define progress bar
  pb <- progress_bar$new(
    format = "Step :current/:total [:bar] :percent in :elapsed",
    total = num_steps,
    clear = FALSE,  
    width = 60
  )
  for(step in 1:num_steps){

    batch <- all_samples[1:min(length(all_samples),batch_size)]
    all_samples <- all_samples[-c(1:length(batch))]

    x.train <- list(Transformer_3Gram=x_1[batch,],
                    Transformer_5Gram=x_2[batch,]
    )
    if(!is.null(factor_vars)){
      x.train <- c(list(FACTORS = x.factors[batch,]),
                   x.train)
    }
    if(oneHot){
      x.train <- c(list(OneHot = as.matrix(cbind(x[batch,],x2[batch,]))),
                   x.train)
    }


    weights.batch <- NULL
    if(useWeights == TRUE){
      weights.batch <- matrix(as.numeric(dat[batch,]$count), nrow=length(batch))
    }

    model %>% train_on_batch(
      x.train,
      sample_weight = weights.batch,
      dat[batch,][["ISCO_target"]]
    )

    pb$tick()
    rm(x.train);gc()
  }

  # evaluate on batch and take averages
  all_samples <- index.valid
  out_eval <- list()
  num_steps <- ceiling(length(index.valid)/batch_size)
  for(step in 1:num_steps){

    batch <- all_samples[1:min(length(all_samples),batch_size)]
    all_samples <- all_samples[-c(1:length(batch))]

    x.valid <- list(Transformer_3Gram=x_1[batch,],
                    Transformer_5Gram=x_2[batch,]
    )
    if(!is.null(factor_vars)){
      x.valid <- c(list(FACTORS = x.factors[batch,]),
                   x.valid)
    }
    if(oneHot){
      x.valid <- c(list(OneHot = as.matrix(cbind(x[batch,],x2[batch,]))),
                   x.valid)
    }

    eval_model <- evaluate(model,
                           x.valid, dat[batch,][["ISCO_target"]],
                           callbacks = callbacks, batch_size = 1000)
    eval_model <- as.data.table(as.list(eval_model))
    eval_model[,N:=length(batch)]

    out_eval <- c(out_eval, list(eval_model))
    rm(x.valid);gc()
  }
  out_eval <- rbindlist(out_eval)
  avg_cols <- colnames(out_eval)
  avg_cols <- avg_cols[avg_cols !="N"]
  eval_model <- out_eval[,sapply(.SD,function(z,N){weighted.mean(z,N)},N=N),.SDcols=c(avg_cols)]

  if(eval_model[[topk]]>current_best){
    current_best <- eval_model[[topk]]
    current_best_m <- m
    keras::save_model_hdf5(model, filepath = file.path(localPath,"model.keras"))
  }
  if(m-current_best_m>(patience-1)){
    break
  }
  gc();
}

# evaluate on test data -----------------------------------------------------
all_samples <- index.test
out_eval_test <- list()
num_steps <- ceiling(length(all_samples)/batch_size)
for(step in 1:num_steps){
  
  batch <- all_samples[1:min(length(all_samples),batch_size)]
  all_samples <- all_samples[-c(1:length(batch))]

  
  x.test <- list(Transformer_3Gram=x_1[batch,],
                  Transformer_5Gram=x_2[batch,]
  )
  if(!is.null(factor_vars)){
    x.test <- c(list(FACTORS = x.factors[batch,]),
                 x.test)
  }
  if(oneHot){
    x.test <- c(list(OneHot = as.matrix(cbind(x[batch,],x2[batch,]))),
                 x.test)
  }
  
  eval_model_test <- evaluate(model,
                         x.test, dat[batch,][["ISCO_target"]],
                         callbacks = callbacks, batch_size = 1000)
  eval_model_test <- as.data.table(as.list(eval_model_test))
  eval_model_test[,N:=length(batch)]
  
  out_eval_test <- c(out_eval_test, list(eval_model_test))
  rm(x.test);gc()
}
out_eval_test <- rbindlist(out_eval_test)
avg_cols <- colnames(out_eval_test)
avg_cols <- avg_cols[avg_cols !="N"]
eval_model_test <- out_eval_test[,sapply(.SD,function(z,N){weighted.mean(z,N)},N=N),.SDcols=c(avg_cols)]

# final results of model performance tested on the test data that was not seen during training
eval_model_test

