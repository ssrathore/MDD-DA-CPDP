
# Eval

def evaluate( encoder, classifier, data_loader ):


  pred_list = [ ]
  label_list = [ ]


  with torch.no_grad():

    for ( features, labels ) in data_loader:

      pred_eval =  classifier( encoder( features ) )

      pred = pred_eval.max( 1 )[ 1 ]

      pred_list.extend( pred )
      label_list.extend( labels )


  f1 = f1_score( label_list, pred_list )
  recall = recall_score( label_list, pred_list )
  g_mean = geometric_mean_score( label_list, pred_list )
  balanced_accuracy = balanced_accuracy_score( label_list, pred_list )


  print( '-----------------------------' )
  print( f'f1 score = { f1 } ' )
  print( f'recall = { recall } ' )
  print( f'g-mean = { g_mean } ' )
  print( f'balanced_accuracy = { balanced_accuracy } ' )
  print( '------------------------------' )
