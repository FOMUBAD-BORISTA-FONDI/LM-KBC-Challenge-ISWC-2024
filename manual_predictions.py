import json

# Step 1: Load the Dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl('path_to_your_train.jsonl')

# Step 2: Explore the Dataset
print("Sample data from the dataset:")
print(train_data[0])

# Step 3: Manual Prediction
def manual_predict(subject_entity, relation):
    predictions = []
    for item in train_data:
        if item['SubjectEntity'] == subject_entity and item['Relation'] == relation:
            predictions.extend(item['ObjectEntities'])
    return predictions

# Step 4: Evaluate the Manual Predictions
def evaluate_manual_predictions(predictions, ground_truth):
    correct = len(set(predictions) & set(ground_truth))
    total = len(ground_truth)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Test the manual prediction function
subject_entity = "Fields medal"
relation = "awardWonBy"
ground_truth = [
    "June Huh", "Maryna Viazovska", "James Maynard", "Hugo Duminil-Copin",
    "Caucher Birkar", "Alessio Figalli", "Maryam Mirzakhani", "Manjul Bhargava",
    "Artur Ávila", "Maxim Kontsevich", "Vladimir Drinfeld", "Vaughan Jones", 
    "Daniel Quillen", "Vladimir Voevodsky", "Jean-Christophe Yoccoz", 
    "Michael Freedman", "Enrico Bombieri", "Richard Borcherds", "Curtis T. McMullen", 
    "Klaus Roth", "Sergei Novikov", "Efim Zelmanov", "Cédric Villani", 
    "Elon Lindenstrauss", "David Mumford", "Pierre-Louis Lions", 
    "Simon Donaldson", "Pierre Deligne", "William Thurston", "Lars Hörmander", 
    "Stanislav Smirnov", "Kunihiko Kodaira", "Shing-Tung Yau", "Grigory Margulis", 
    "Andrei Okounkov", "Stephen Smale", "Laurent Lafforgue", "Jesse Douglas", 
    "John G. Thompson", "Alain Connes", "Alan Baker", "Terence Tao", 
    "Charles Fefferman", "Jean Bourgain", "Timothy Gowers", "John Milnor", 
    "Ngô Bảo Châu", "Paul Cohen", "Atle Selberg", "Lars Ahlfors", 
    "Laurent Schwartz", "Jean-Pierre Serre", "Shigefumi Mori", 
    "Michael Atiyah", "Edward Witten", "Heisuke Hironaka", "Akshay Venkatesh", 
    "Wendelin Werner", "René Thom", "Grigori Perelman", "Peter Scholze", 
    "Martin Hairer", "Gerd Faltings", "Alexander Grothendieck"
]

predictions = manual_predict(subject_entity, relation)
print(f"Predictions for {subject_entity} ({relation}): {predictions}")

accuracy = evaluate_manual_predictions(predictions, ground_truth)
print(f"Accuracy: {accuracy}")

# Step 5: Refine Your Predictions
def refined_manual_predict(subject_entity, relation):
    predictions = []
    for item in train_data:
        if item['SubjectEntity'] == subject_entity and item['Relation'] == relation:
            predictions.extend(item['ObjectEntities'])
        # Add more refined logic here if necessary
    return predictions

refined_predictions = refined_manual_predict(subject_entity, relation)
print(f"Refined Predictions for {subject_entity} ({relation}): {refined_predictions}")

refined_accuracy = evaluate_manual_predictions(refined_predictions, ground_truth)
print(f"Refined Accuracy: {refined_accuracy}")
