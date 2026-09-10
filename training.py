import tensorflow as tf 
def train_model(model, train_generator, test_images, test_labels, epochs):
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Set up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model using a generator
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        validation_data=(test_images, test_labels),
                        callbacks=[early_stopping])
    
    return history
