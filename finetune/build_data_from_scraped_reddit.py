import json

# File paths (modify as needed)
submissions_file = "datasets\raw_data\AskAGerman_submissions"
comments_file = "datasets\raw_data\AskAGerman_comments"
output_file = "datasets\askagerman_filtered1000.json"

# Step 1: Read submissions and store in a dictionary
submissions = {}
with open(submissions_file, "r", encoding="utf-8") as f:
    for line in f:
        submission = json.loads(line)
        selftext = submission.get("selftext", "").strip()
        title = submission.get("title", "").strip()
        distinguished = submission.get("distinguished", "")
        score = submission.get("score", 1)

        
        if selftext and selftext.lower() != "[deleted]" and distinguished != "moderator" and score > 50:
            submissions[submission["id"]] = title + ". \n\n" + selftext

# Step 2: Process comments and match with submissions
qa_pairs = []
with open(comments_file, "r", encoding="utf-8") as f:
    for line in f:
        comment = json.loads(line)
        parent_id = comment.get("parent_id", "").split("_")[-1]  # Remove prefix
        body = comment.get("body", "").strip()
        score = comment.get("score", 1)
        is_submitter = comment.get("is_submitter")
        
        if (parent_id in submissions and body and 
            body.lower() != "[removed]" and len(body.split()) > 1 and score > 50 and is_submitter==False):
            qa_pairs.append({"input": submissions[parent_id], "output": body})

# Step 3: Save the extracted Q&A pairs
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

print(f"Extracted {len(qa_pairs)} Q&A pairs and saved to {output_file}")