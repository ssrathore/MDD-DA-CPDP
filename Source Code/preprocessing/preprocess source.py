
# pre - process source


source = pd.read_csv( '/content/Mylyn.csv' )

labels = source[ 'class' ].values
features = source.drop( [ 'class' ], axis = 1 ).values


features_source_train, features_source_test, labels_source_train, labels_source_test = train_test_split( features, labels, test_size = 0.2 )


scaler = StandardScaler()
features_source_train = scaler.fit_transform( features_source_train )
features_source_test = scaler.transform( features_source_test )


smote = SMOTE()
features_source_train, labels_source_train = smote.fit_resample( features_source_train, labels_source_train )


features_source_train = torch.Tensor( features_source_train )
features_source_test = torch.Tensor( features_source_test )

labels_source_train = torch.Tensor( labels_source_train )
labels_source_test = torch.Tensor( labels_source_test )


dataset_source_train = utils.TensorDataset( features_source_train, labels_source_train )
dataset_source_test = utils.TensorDataset( features_source_test, labels_source_test )
