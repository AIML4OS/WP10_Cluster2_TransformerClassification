#helper functions for tokenization etc

library(stringr)
library(stringdist)
library(stopwords)
library(keras)
library(tokenizers)
library(stringi)

#Texte vereinfachen
#this function takes the text column of the data and cleans it
simplify_text <- function(datcol){
  patterns <- c("\\*inn\\b","[:|\\*|_|/]in","/-?in\\b",
                "\\(?([d|x][/|\\||,])?m\\.?[/|\\||,]\\??[w|f]\\.?\\)?",
                "\\(?([d|x][/|\\||,])?[w|f]\\.?[/|\\||,]m\\.?\\)?",
                "\\(?[w|f]\\.?[/|\\||,]m\\.?[/|\\||,][d|x]\\.?\\)?",
                "\\(?m.?[/|\\||,][w|f|e|m]\\.?[/|\\||,][d|x]\\.?\\)?",
                "\\(?m\\.?[/|\\||,][d|x]\\.?[/|\\||,][w|f]\\.?\\)?",
                "erin\\b","/\\s?-?frau\\b","\\(?junior\\)?",
                "\\(?senior\\)?\\s?",
                "frau/?-?(\\s)?mann",
                "\\(m_f\\)",
                #"",
                #"frau\\b",
                "frau:mann",
                "frau / mann\\b",
                "frau\\b",#wortfrau zu wortmann
                "d\\)",
                "/.\\*mann\\b",
                "\\-\\)",
                "\\(Sr\\.\\)",
                "\\(jr\\.\\)",
                "(\\()?Pflicht(\\))?-?",
                "/[-|\\*]?r\\b", "-9","9",
                "\\(in\\)",
                "x\\)",
                "-mann\\b",
                ",","\\)","\\(",
                "/-?köchin\\b",
                "/-?koch\\b",
                #"\\s?[/|\\*]\\s?-?\\s?ärztin\\b", "ärztin\\s?[/|\\*|-]\\s?","[a-z]ärztin\\b",
                "/[r|n|m]\\b",
                " w\\b", '""',"-2\\s?","-1\\s?",
                "^\\s?[-|\\.|\\>|\\<|'|&|/]+\\s?",
                "1st","2nd","3rd",
                "[^|/]s?[0-9]+\\s?[\\.|\\-|\\%]?[>|\\s]?",
                "^[\\.|-]\\s?",
                "bursche/?-?madchen\\b",
                "[_|\\*][r|n]\\b",
                "/en\\b",
                "[-|\\+]",
                "human resources",
                "public relations",
                "^gastwirt\\b",
                "\\s?/\\s?",
                "\\s?&\\s?",
                "[:|;|\\*|~|\\?|=|\\<|\\>|%|\\.|\\$|€|_|µ|\\^|°|\\||,|\\-]",
                "madchen/?-bursche\\b",
                "(\\w+)@(\\w+)",
                "–",
                "[0-9][\\s|\\.|g\\s|\\b]", #g wegen 5g
                "[\\[|\\]]",
                "^praktikant\\b",
                "\\(?lehrling(e)?\\)?\\b", #es gibt keinen isco code für lehrlinge
                "!+",
                " [d|u] ",
                "jr\\.(\\s)?",
                "é","è")
  replacements <- c("","","","","","","","","er","","","","mann",#"",#"",
                    "mann","mann", "mann", "mann","","","","","","","","","","","","",
                    "","","","","",
                    #"","","arzt",
                    "","","","","",
                    "","first","second","third","","",
                    "madchen","",""," ",
                    "hr","pr","gastronom",
                    " ","", "",
                    "madchen",
                    "","","","",
                    "","",
                    "","","",
                    "e","e"
                    
  )
  
  
  #alle umlaute zu vokalen
  umlaut <- c("Ä","Ö","Ü","ä","ö","ü","ß")
  kodierung <- c("a","o ","u","a","o","u","ss")
  datcol <- stri_replace_all_regex(datcol, umlaut, kodierung, vectorize_all = FALSE)
  
  # datcol <- apply(matrix(datcol),1,function(x) gsub("(\\w+)\\s?[/|:|\\s|*]\\s?\\1in",stri_split_regex(x,"[/|:]")[[1]][1],x))
  # datcol <- apply(matrix(datcol),1,function(x) gsub("(\\w+)in\\s?[/|:|\\s|*]\\s?\\1e?",stri_split_regex(x,"[/|:]")[[1]][2],x))
  # datcol <- apply(matrix(datcol),1,function(x) gsub("[a-z]{3,}innen\\b",stri_split_regex(x,"innen")[[1]][1],x))
  datcol <- sapply(datcol,function(x){
    char_replace <- stri_split_regex(x,"[/|:]")[[1]][1]
    if(is.na(char_replace)){
      return(x)
    }
    return(gsub("(\\w+)\\s?[/|:|\\s|*]\\s?\\1in",char_replace,x))
  })
  datcol <- sapply(datcol,function(x){
    char_replace <- stri_split_regex(x,"[/|:]")[[1]][2]
    if(is.na(char_replace)){
      return(x)
    }
    return(gsub("(\\w+)in\\s?[/|:|\\s|*]\\s?\\1e?",char_replace,x))
  })
  datcol <- sapply(datcol,function(x){
    char_replace <- stri_split_regex(x,"innen")[[1]][1]
    if(is.na(char_replace)){
      return(x)
    }
    return(gsub("[a-z]{3,}innen\\b",char_replace,x))
  })
  
  #stopwords removal
  stopwords <- stopwords(language = "de", source =  "snowball")
  stopwords <- stopwords[!stopwords%in%c("nicht")]
  stopwords <- paste0("\\b",stopwords,"\\b")
  stopwords <- stri_replace_all_regex(stopwords,umlaut,kodierung,vectorise_all = FALSE)
  datcol <- stri_replace_all_regex(datcol, stopwords, "", vectorize_all = FALSE)
  
  
  datcol <- stri_replace_all_regex(datcol, patterns, replacements, vectorize_all = FALSE)
  datcol <- trimws(stri_replace_all_regex(datcol,"\\s+"," "))
  #dat$Benennung <- trimws(stri_replace_all_regex(dat$Benennung, patterns, replacements, vectorize_all = FALSE))
  return(datcol)
}


#maxlen: max. Anzahl an Token die für einen input string verwendet werden
setupEmbedding2 <- function(dat1, string_col = "STRING_CLEAN2", ngram = 3, roll = FALSE, 
                            keep_spaces = TRUE, space_token = TRUE, tok=NULL,
                            maxlen=NULL, left_pad=FALSE){
  
  dat <- copy(dat1)
  
  if(!is.data.table(dat)){
    dat <- as.data.table(dat)
  }
  
  if(keep_spaces == FALSE){
    dat[,c(string_col):=gsub("\\s","",get(string_col))]
  }
  
  dat[,ID:=1:nrow(dat)]
  x <- dat[,.(Word=unlist(str_split(get(string_col),pattern=" "))),by=.(ID)]
  x[,Position:=1:.N,by=.(ID)]
  x_gram <- x[,.(Word_ngram=split_word(unlist(.BY),ngram = ngram, roll = roll)),by=.(Word)]
  
  if(keep_spaces == TRUE & space_token == TRUE){
    x_gram <- rbind(x_gram,
                    data.table(Word="_space_",Word_ngram="_space_"))
  }
  
  x_gram[,Word_order:=1:.N,by=.(Word)]
  
  if(keep_spaces == TRUE & space_token == TRUE){
    x[,filter_length:=.N>1,by=.(ID)]
    x_space <- x[filter_length == TRUE,.(Position = (Position+.5)[-.N]), by = .(ID)]
    x_space[,Word:="_space_"]
    x <- rbind(x, x_space,use.names = TRUE, fill = TRUE)
  }
  
  x <- x[x_gram,on=.(Word), nomatch = NULL, allow.cartesian = TRUE]
  setorder(x,ID,Position,Word_order)
  
  if(is.null(tok)){
    x[order(Word_ngram),TOKEN:=.GRP,by=.(Word_ngram)]
    num_words <- x[,uniqueN(Word_ngram)] + 1
    tok <- unique(x[,.(Word_ngram,TOKEN)])
  }else{
    x[tok,TOKEN:=TOKEN,on=.(Word_ngram)]
    num_words <- tok[,uniqueN(TOKEN)] + 1
  }
  
  x[,help_position:=1:.N,by=.(ID)]
  x[help_position==1 & is.na(TOKEN),TOKEN:=0]
  #x <- x[!is.na(TOKEN)]
  unknown_token <- 0
  x[is.na(TOKEN),TOKEN:=unknown_token] #add unknown token
  
  out_idx <- dcast(x,ID~help_position,value.var = "TOKEN", fill=0)
  out_idx[,ID:=NULL]
  out_idx <- as.matrix(out_idx)
  
  #shorten emb to maxlen
  if(!is.null(maxlen)){
    if(maxlen<dim(out_idx)[2]){
      out_idx <- out_idx[,1:maxlen]
    }else if(maxlen>dim(out_idx)[2]){
      pad <- matrix(0,ncol=maxlen-ncol(out_idx),nrow=nrow(out_idx))
      out_idx <- cbind(out_idx,pad)
    }
  }
  
  if(left_pad){
    out_idx <- t(apply(out_idx,1,left_padding))
  }
  
  tok=rbind(tok,data.table("Word_ngram"="_pad_","TOKEN"=0)) #add padding token
  return(list(out_idx, num_words=num_words, tok = tok))
}


split_word <- function(x_word, ngram=3, roll = FALSE){
  
  if(nchar(x_word)<=ngram){
    return(x_word)
  }
  
  x_word <- unlist(str_split(x_word, pattern=""))
  
  if(roll == FALSE){
    subset_indices <- seq(1,length(x_word),by=ngram)
    x_word <- sapply(subset_indices,function(start, x_word, ngram){
      out <- x_word[start:min(length(x_word),(start+ngram-1))]
      paste(out,collapse="")
    },x_word=x_word,ngram=ngram)
  }else{
    x_word <- paste(x_word, collapse = " ")
    x_word <- tokenize_ngrams(x_word,n=ngram,ngram_delim="", simplify = TRUE)
    x_word <- unlist(x_word)
  }
  
  # x_word <- paste(x_word,collapse=" ")
  return(x_word)
  
}

# help function to add cols of 0s to matrix
add0Cols <- function(mat,ncol){
  
  fill0 <- ncol-ncol(mat)
  if(fill0>0){
    mat <- cbind(mat,matrix(0,ncol=fill0,nrow=nrow(mat)))
  }else if(fill0<0){
    mat <- mat[,1:ncol,drop=FALSE]
  }
  return(mat)
}

# transform token matrix into one hot encode feature matrix
setupOneHot <- function(x, id_train = NULL, col_position = NULL){
  
  # build one hot encoding matrix from tokenized inputs
  # one hot encode 
  n_rec <- nrow(x)
  # if(!is.null(id_train)){
  #  n_rec <- length(id_train) 
  # }
  
  x <- as.data.table(x)
  m_vars <- copy(colnames(x))
  x[,ID:=.I]
  x <- melt(x, id.vars="ID", measure.vars = m_vars, value.name = "Token")
  x <- x[Token!=0]
  
  if(!is.null(col_position)){
    x[col_position,IDF:=IDF,on=.(Token)]
  }else{
    if(!is.null(id_train)){
      x[ID %in% id_train,IDF:=log(n_rec/uniqueN(ID)), by=.(Token)] # for training use only training subset
    }else{
      x[,IDF:=log(n_rec/uniqueN(ID)), by=.(Token)] # for training use only training subset
    }
    x[,IDF:=IDF[!is.na(IDF)][1],by=.(Token)]
  }
  x[is.na(IDF),IDF:=log(n_rec/1)]
  x[,TF:=as.numeric(.N),by=.(ID,Token)]
  x[,TF:=TF/max(TF),by=.(ID)]
  x[,TFIDF:=TF * IDF]
  x <- unique(x, by=c("Token","ID"))
  # n_rows <- x[,uniqueN(ID)]
  if(!is.null(col_position)){
    x[col_position,Token_position:=Token_position, on=.(Token)]
    x <- x[!is.na(Token_position)]
  }else{
    x[,Token_position:=.GRP,by=.(Token)] # relabel tokens
    col_position <- unique(x[,.(Token, Token_position, IDF)])
  }
  n_cols <- max(col_position[["Token_position"]]) 
  # gc(x)
  x <- Matrix::sparseMatrix(i=x[["ID"]], j=x[["Token_position"]],x=x[["TFIDF"]],dims = c(n_rec,n_cols))
  
  return(list(x, col_position))
  
}


#moves padding from right to left
left_padding <- function(vec){
  pos_zero <- which(vec==0)[1]
  if(is.na(pos_zero)){
    return(vec)
  }else{
    num_zero <- length(vec)-which(vec==0)[1]+1
    temp <- vec[1:pos_zero-1]
    out <- c(rep(0,num_zero),temp)
    return(out)
  }
}


metric_top_10_categorical_accuracy <-
  custom_metric("top_10_categorical_accuracy", function(y_true, y_pred) {
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 10)
  })
metric_top_5_categorical_accuracy <-
  custom_metric("top_5_categorical_accuracy", function(y_true, y_pred) {
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 5)
  })



# add variabls to alphabetikum
add_vars_alphabetikum <- function(looktab, dat, add_vars, miss_values, target_var){
  
  add_vars_lfn <- dat[!is.na(Laufnummer) & Laufnummer !="",.N,c(add_vars, "Laufnummer")]
  add_vars_lfn[,p:=N/sum(N),by=.(Laufnummer)]
  setorder(add_vars_lfn, Laufnummer, -p)
  add_vars_lfn[,p_cs := cumsum(p),by=.(Laufnummer)]
  add_vars_lfn <- add_vars_lfn[p_cs<=0.975 | p>=0.975]
  add_vars_lfn[,c("p","p_cs"):=NULL]
  looktab_extra <- merge(looktab,add_vars_lfn,by="Laufnummer", all.x=TRUE)
  
  add_vars_tab <- dat[,.N,c(add_vars, target_var)]
  setorderv(add_vars_tab,c(target_var,"N"), order(1,-1))
  add_vars_tab[,p_cs:=cumsum(N/sum(N)),by=c(target_var)]
  add_vars_tab[,filter:=p_cs>=1/3,by=c(target_var)]
  add_vars_tab[,filter:=filter | N==max(N),by=c(target_var)]
  add_vars_tab[filter==FALSE,.N,by=c(target_var)]
  add_vars_tab[,mean(filter)]
  
  looktab_extra2 <- merge(looktab[Laufnummer %in% looktab_extra[is.na(get(add_vars[1]))]$Laufnummer],add_vars_tab[filter==TRUE],by=target_var,all.x=TRUE, allow.cartesian = TRUE)
  looktab_extra2[,c("filter","p_cs"):=NULL]
  looktab_extra <- rbind(looktab_extra[!is.na(get(add_vars[1]))], looktab_extra2, use.names = TRUE)
  looktab_extra[is.na(get(add_vars[1])), c(add_vars):=as.list(miss_values)]
  
  # not needed anymore
  # looktab_uml <- looktab_extra[Text %ilike% "ä|ü|ö"]
  # looktab_uml[,Text:=gsub("ä|ü|ö","",Text,ignore.case = TRUE)]
  looktab <- looktab_extra # rbind(looktab_extra,looktab_uml)
  looktab[,ID:=as.character(.I)]
  looktab[,Source:="Dictionary"]
  looktab[,count:=N/sum(N),by=.(Laufnummer)]
  looktab[is.na(count), count:=1:.N,by=.(Laufnummer)]
  
  return(looktab)
}

applyStringDistance <- function(x,dict_df,method=c("osa","qgram"),q=2,return_lfn=F){
  #dict_df is a datatable with col "Benennung" for economic acticity and "NACE08" for NACE code
  
  
  #if(all(is.na(dict))){
  #  return(data.table(STRING=NA_character_,ProbabilityString=NA_real_))
  #}
  n <- nrow(dict_df)
  m <- length(method)
  distsim <- matrix(0, nrow = n, ncol = m)
  
  dict <- dict_df$Benennung
  if(return_lfn){
    codes <- dict_df$Laufnummer
  }else{
    if("Code08"%in%names(dict_df)){
      codes <- dict_df$Code08
    }else{
      codes <- dict_df$NACE08
    }
  }
  
  
  
  # calculate string distance for each method
  for (i in 1:m) {
    
    distsim_i <- stringsim(x, dict, method = method[i], q = q)
    distsim[, i] <- distsim_i
  }
  # apply mean over different methods - all combinations
  
  distsim <- matrixStats::rowMeans2(distsim)
  
  # position of closest String
  closestStringPos <- which(distsim==max(distsim))
  if(length(closestStringPos)>1){
    closestStringPos <- sample(closestStringPos,1)
  }
  
  # get closest String
  closestString <- dict[closestStringPos]
  
  # get Code of closest String
  closestStringCode <- codes[closestStringPos]
  
  # get probability of closest String
  closestStringProb <- distsim[closestStringPos]
  
  
  # build output
  output <- data.table(text=x,
                       Prediction=closestString,
                       Code_Pred=closestStringCode,
                       Prob=closestStringProb)
  return(output)
}







# vectorized cosine similarity
cosine_similarity_matrix <- function(A, B) {
  # ensure the input matrices have the same dimensions
  if (ncol(A) != ncol(B)) {
    stop("The number of columns in A must be equal to the number of columns in B.")
  }
  # Compute the dot products between all vectors in A and B
  dot_products <- A %*% t(B)  # Matrix multiplication A x B^T
  
  # Compute the norms of all vectors in A and B
  norm_A <- sqrt(rowSums(A * A))  # Norm of each row in A (m x 1)
  norm_B <- sqrt(rowSums(B * B))  # Norm of each row in B (p x 1)
  
  # Normalize the dot products to get the cosine similarities
  # We need to divide each element in the dot product matrix by the product of the corresponding norms
  similarity_matrix <- sweep(dot_products, 1, norm_A, "/")  # Divide each row by norm_A
  similarity_matrix <- sweep(similarity_matrix, 2, norm_B, "/")  # Divide each column by norm_B
  
  most_similar_indices <- apply(similarity_matrix, 1, which.max)
  similarities <- apply(similarity_matrix,1,max)
  
  return(list(most_similar_ind=most_similar_indices,similarities=similarities))
}



# Define the function to find the most similar string
## query string: string to calculate cosine sim for
## dict_embeddings:  precomputed embeddings of dictionary
## string_list: c() of text corresponding to dict_embeddings
## model_sim: loaded model from huggingface sentence transformers
## return_prob: return cosine similarity
## parallel: compute query embeddings in parallel (Vorsicht deadlocks mit tokenizer)
# get_most_similar <- function(query_string, 
#                              dict_embeddings,
#                              string_list,
#                              model_sim,
#                              return_prob=F,
#                              num_cores = 3) {
#   if(length(string_list)!=nrow(dict_embeddings)){
#     stop("String_list and dict_embeddings must be of same length!")
#   }
#   
#   # Encode the query string and the list of strings
#   
#   # parallel sessions erst ab ca. 180 input texten sinnvoll (wegen overhead)
#   if(length(query_string)>180){
#     chunked_x <- split(query_string, ceiling(seq_along(query_string) / (length(query_string) / num_cores)))
#     
#     plan(multisession, workers = num_cores)
#     
#     # Apply the encoding in parallel
#     result <- future_lapply(chunked_x,  function(x) {
#       sentence_transformers <- import("sentence_transformers")
#       model_sim <- sentence_transformers$SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#       model_sim$encode(x)
#     },future.seed = T)
#     
#     # Combine the results
#     query_embedding <- do.call(rbind, result)
#     
#     # Reset plan to sequential after processing
#     plan(sequential)
#     
#   }else{
#     query_embedding <- model_sim$encode(query_string)
#   }
#   
#   # if query embedding is 1dim vecotr -> matrix with one row
#   if(length(dim(query_embedding))==1){
#     dim(query_embedding) <- c(1,dim(query_embedding))
#   }
#   
#   # Compute cosine similarities
#   # similarities <- apply(dict_embeddings,1, function(x) {
#   #   cosine_similarity(query_embedding, x)
#   # })
#   
#   # Find the index of the maximum similarity
#   most_similar <- cosine_similarity_matrix(query_embedding,dict_embeddings)
#   most_similar_idx <- most_similar$most_similar_ind
#   similarities <- most_similar$similarities
#   
#   if(return_prob){
#     return(list(string=string_list[most_similar_idx],
#                 probs=similarities))
#   }else{
#     # Return the most similar string
#     return(string_list[most_similar_idx])
#   }
# }
# 
# 
# find_closest_lnr_mod <- function(query_string,dictionary,model,return_prob=F){
#   dict_embeddings <- dictionary[,5:ncol(dictionary)] #nur embedding cols 
#   string_list <- dictionary$Text_clean
#   closest_string <- get_most_similar(query_string=query_string,dict_embeddings=dict_embeddings,
#                                      string_list=string_list,model=model,
#                                      return_prob=return_prob)
#   #res <- dictionary[Text_clean==closest_string,Laufnummer][1]
#   res <- as.integer(sapply(closest_string, function(x) dictionary[Text_clean == x,Laufnummer][1]))
#   return(res)
# }

