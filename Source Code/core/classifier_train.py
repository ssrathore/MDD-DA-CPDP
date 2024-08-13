
def pre_train( encoder, classifier, data_loader ):

  encoder.train()
  classifier.train()

  optimizer = optim.Adam( params = list( encoder.parameters() ) + list( classifier.parameters() ), lr = 0.001 )
  loss_fn = nn.CrossEntropyLoss()

  min_loss = 1000000
  losses = []


  ## Training the network

  for epoch in range( n_epochs_pre ):

    print( f' Current Epoch = { epoch + 1 } ' )
    for step, ( features, labels ) in enumerate( data_loader ):

      labels = labels.type( torch.LongTensor )


      features = features.to( device )
      labels = labels.to( device )


      optimizer.zero_grad()

      preds = classifier( encoder( features ) )
      loss = loss_fn( preds, labels )


      loss.backward()
      optimizer.step()

      if step % 5 == 0:
        print( f' Current Epoch = { epoch + 1 }, Current Loss = { loss }')


    losses.append( loss.detach().numpy() )
    if loss < min_loss:
      min_loss = loss
      torch.save( encoder.state_dict(), 'best_encoder.pt' )
      torch.save( classifier.state_dict(), 'best_classifier.pt' )



  plt.plot( losses )
  plt.xlabel( ' Epochs ' )
  plt.ylabel( ' Loss values ' )
  plt.title( ' Classifier loss on pre-training ' )


  return encoder, classifier
