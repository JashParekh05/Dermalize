from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a dense layer with the number of classes for your specific problem
predictions = Dense(6, activation='softmax')(x)

# This is your new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model if needed
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model, including architecture and weights
model.save('/Users/jashparekh/Desktop/Dermalize/Backend/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
