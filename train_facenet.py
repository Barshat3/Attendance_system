import tensorflow as tf
from facenet_model import FaceNet_InceptionResNet, triplet_loss
from data_loader import load_images
from triplet_generator import generate_triplet_batch

# Set dataset path and training parameters
DATASET_PATH = '/home/barshat/Desktop/Attendence System/dataset' 
epochs = 10
batch_size = 16
steps_per_epoch = 100
embedding_dim = 128

# Load the data
data = load_images(DATASET_PATH)

# Initialize the model
facenet_model = FaceNet_InceptionResNet(input_shape=(160, 160, 3), embedding_dim=embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Custom training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step in range(steps_per_epoch):
        # Generate a batch of triplets
        anchors, positives, negatives = generate_triplet_batch(data, batch_size)
        
        with tf.GradientTape() as tape:
            # Pass each image through the model to get embeddings
            anchor_embeddings = facenet_model(anchors, training=True)
            positive_embeddings = facenet_model(positives, training=True)
            negative_embeddings = facenet_model(negatives, training=True)

            # Combine embeddings for triplet loss calculation
            embeddings = [anchor_embeddings, positive_embeddings, negative_embeddings]
            loss = triplet_loss(None, embeddings)  # Calculate the loss using the embeddings

        # Compute gradients and apply updates
        gradients = tape.gradient(loss, facenet_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, facenet_model.trainable_variables))
        
        print(f"Step {step + 1}/{steps_per_epoch}, Loss: {loss.numpy()}")

# Save the trained model
facenet_model.save("/home/barshat/Desktop/Attendence System/models/facenet_inception_resnet.h5")