rm ./submission/submission.zip
SUBMISSION_TRACK=fincrime make pack-submission
# SUBMISSION_TRACK=fincrime SUBMISSION_TYPE=federated make test-submission
SUBMISSION_TRACK=fincrime SUBMISSION_TYPE=centralized make test-submission