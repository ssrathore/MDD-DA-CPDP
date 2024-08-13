
def adversarial_training( source_encoder, target_encoder, discriminator, source_data_loader, target_data_loader ):


  ## Setup

  target_encoder.train()
  discriminator.train()


  loss_func = nn.CrossEntropyLoss()
  target_optimizer = optim.Adam( target_encoder.parameters(), lr = 0.01 )
  discriminator_optimizer = optim.Adam( discriminator.parameters(), lr = 0.0004 )

  len_data_loader = min( len( source_data_loader ), len( target_data_loader ) )


  min_d_loss = float( ' inf ' )
  min_te_loss = float( ' inf ' )


  ## Training

  for epoch in range( n_epochs ):

    data_zip = enumerate( zip( source_data_loader, target_data_loader ) )

    for step, ( ( features_source_pre, _ ), ( features_target_pre, _ ) ) in data_zip:



      # Train Discriminator


      features_source_pre = features_source_pre.to( device )
      features_target_pre = features_target_pre.to( device )


      discriminator_optimizer.zero_grad()


      features_source = source_encoder( features_source_pre )
      features_target = target_encoder( features_target_pre )
      features_concat = torch.cat( ( features_source, features_target ), 0 )


      pred_concat = discriminator( features_concat )
      pred_concat = pred_concat.squeeze()


      label_source = torch.ones( features_source.size( 0 ) ).long().to( device )
      label_target = torch.zeros( features_target.size( 0 ) ).long().to( device )
      label_concat = torch.cat( ( label_source, label_target ), 0 )


      loss_discriminator = loss_func( pred_concat, label_concat )
      loss_discriminator.backward()


      discriminator_optimizer.step()


      ## Train target encoder

      discriminator_optimizer.zero_grad()
      target_optimizer.zero_grad()


      features_target = target_encoder( features_target_pre )

      pred_target = discriminator( features_target )
      #pred_target = pred_target.squeeze( 0 )

      label_target = torch.ones( features_target.size( 0 ) ).long().to( device )
      label_target = label_target.type( torch.LongTensor )


      loss_target = loss_func( pred_target, label_target )
      loss_target.backward()


      target_optimizer.step()


      if step % 5 == 0:
        print( f' Current Epoch = { epoch }, Discriminator Loss = { loss_discriminator }, Target Encoder Loss = { loss_target } ' )


    #print( 'this will print after every epoch ' )
    if loss_discriminator < min_d_loss and epoch > 20 and loss_target < 1:
      min_d_loss = loss_discriminator
      print( 'saved disc here ' )
      torch.save( discriminator.state_dict(), 'best_discriminator.pt' )

    if loss_target < min_te_loss and epoch > 20 and loss_discriminator < 1:
      min_te_loss = loss_target
      print( 'saved te here ' )
      torch.save( target_encoder.state_dict(), 'best_target_encoder.pt' )


  return target_encoder
