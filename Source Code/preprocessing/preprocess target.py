
# pre - process target

target = pd.read_csv( '/content/PDE.csv' )

labels = target[ 'class' ].values
features = target.drop( [ 'class' ], axis = 1 ).values


features_target_train, features_target_test, labels_target_train, labels_target_test = train_test_split( features, labels, test_size = 0.2 )


scaler2 = StandardScaler()
features_target_train = scaler2.fit_transform( features_target_train )
features_target_test = scaler2.transform( features_target_test )


smote = SMOTE()
features_target_train, labels_target_train = smote.fit_resample( features_target_train, labels_target_train )


features_target_train = torch.Tensor( features_target_train )
features_target_test = torch.Tensor( features_target_test )

labels_target_train = torch.Tensor( labels_target_train )
labels_target_test = torch.Tensor( labels_target_test )


dataset_target_train = utils.TensorDataset( features_target_train, labels_target_train )
dataset_target_test = utils.TensorDataset( features_target_test, labels_target_test )
