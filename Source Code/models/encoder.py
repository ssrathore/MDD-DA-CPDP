# Encoder model

class Encoder( nn.Module ):
  ''' Encoder will be used to generate important features from the Data '''

  def __init__( self ):
    super().__init__()


    self.fc1 = nn.Linear( 61, 100 )
    self.fc2 = nn.Linear( 100, 200 )
    self.fc3 = nn.Linear( 200, 400 )
    self.fc_final = nn.Linear( 400, 500 )

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout()


  def forward( self, features_in ):


    x = self.relu( self.fc1( features_in ) )
    x = self.relu( self.fc2( x ) )
    x = self.dropout( x )
    x = self.relu( self.fc3( x ) )

    features = self.fc_final( x )


    return features
