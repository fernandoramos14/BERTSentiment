# Save the trained SVM model to a file
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("SVM model saved to 'svm_model.pkl'")
