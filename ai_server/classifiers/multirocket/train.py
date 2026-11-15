"""MultiRocket Stack 배포 번들 학습. 실제 절차는 research/c_part_deploy_train.py 에 있다.

    python -m ai_server.services.train --data research/output/c_part_cv5_v1 \
      --output research/output/c_part_deploy_multirocket --output-path models/multirocket_stack_classifier.joblib
"""
from research.c_part_deploy_train import main

if __name__ == "__main__":
    main()
