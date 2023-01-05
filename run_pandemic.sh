rm ./submission/submission.zip
SUBMISSION_TRACK=pandemic make pack-submission
# SUBMISSION_TRACK=pandemic SUBMISSION_TYPE=federated make test-submission
SUBMISSION_TRACK=pandemic SUBMISSION_TYPE=centralized make test-submission