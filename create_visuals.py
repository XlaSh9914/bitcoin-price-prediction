from keras.models import load_model
from keras.utils import plot_model

# Load the saved neural network model
model = load_model('neural_network_model.h5')

# Save the model architecture as a PNG image with enhanced options
plot_model(model, to_file='neural_network_architecture.png', 
           show_shapes=True,        # Show shape of layers
           show_layer_names=True,   # Show layer names
           rankdir='TB',            # Direction of the graph
           dpi=300)                 # Increase the resolution

print("Neural network architecture saved as 'neural_network_architecture.png'.")