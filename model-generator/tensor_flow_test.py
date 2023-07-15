import tensorflow as tf

# Define model
class M(tf.keras.Model):
    def __init__(self):
        super(M, self).__init__()
        self.fc1 = tf.keras.layers.Dense(10, input_shape=(20,))
        self.relu = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Dense(5)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a model instance
model = M()

# Compile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Generate input sample and dummy labels
x = tf.random.normal((1, 20))
y = tf.constant([1])

# Train the model with the dummy data
# In a real scenario, you should train with your actual data
model.fit(x, y, epochs=1)

# Save the model to a SavedModel format
model.save("model_simple_tf", save_format='tf')
