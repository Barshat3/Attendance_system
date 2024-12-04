import random
import numpy as np

def generate_triplet(data):
    # Filter out individuals with fewer than two images
    eligible_persons = [person for person, images in data.items() if len(images) >= 2]
    if not eligible_persons:
        return None, None, None

    # Randomly select a person for anchor and positive
    person_name = random.choice(eligible_persons)
    positive_images = data[person_name]

    # Select anchor and positive images for the same person
    anchor_img, positive_img = random.sample(positive_images, 2)

    # Select a negative image from a different person
    negative_person = random.choice([p for p in data.keys() if p != person_name])
    negative_img = random.choice(data[negative_person])

    return anchor_img, positive_img, negative_img

def generate_triplet_batch(data, batch_size=32):
    anchors, positives, negatives = [], [], []

    for _ in range(batch_size):
        # Generate a single triplet
        anchor, positive, negative = generate_triplet(data)
        
        if anchor is not None and positive is not None and negative is not None:
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

    # Check if any triplets were generated
    if len(anchors) == 0:
        raise ValueError("No valid triplets generated. Ensure dataset has individuals with at least two images each.")

    return np.array(anchors), np.array(positives), np.array(negatives)
