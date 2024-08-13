# Base Classifier

class Classifier( nn.Module ):
  """ The classifier will be used to classify the features generated from the Encoder """

  def __init__( self ):
    super().__init__()

    self.fc = nn.Linear( 500, 2 )
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout()

  def forward( self, features_in ):

    output = self.dropout( self.relu( features_in ) )
    output = self.fc( output )

    return output
