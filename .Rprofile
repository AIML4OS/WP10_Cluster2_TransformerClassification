source("renv/activate.R")

setHook('rstudio.sessionInit', function(newSession) {
 if (newSession)
  {
    renv::restore(prompt = FALSE)
  }
}, action = 'append')



setHook('rstudio.sessionInit', function(newSession) {
 if (newSession)
  {
    rstudioapi::navigateToFile('/home/onyxia/work/WP10_Cluster2_TransformerClassification/exercises/exercise1.qmd')
  }
}, action = 'append')


