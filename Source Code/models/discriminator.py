
# Domain Discriminator

class Discriminator( nn.Module ):
  """ The discriminator will be used to distinguish between the source and target data """

  def __init__( self ):
    super().__init__()

    self.fc1 = nn.Linear( 500, 250 )
    self.fc2 = nn.Linear( 250, 125 )
    self.fc3 = nn.Linear( 125, 2 )
    self.relu = nn.ReLU()
    self.logsoftmax = nn.LogSoftmax( )

  def forward( self, input ):

    x = self.relu( self.fc1( input ) )
    x = self.relu( self.fc2( x ) )
    output = self.logsoftmax( self.fc3( x ) )

    return output
