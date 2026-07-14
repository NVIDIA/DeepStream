#!/bin/bash
cat <<EOF

===============================
   DeepStreamSDK $DS_VERSION
===============================

*** LICENSE AGREEMENT ***
By using this software you agree to fully comply with the terms and conditions
of the License Agreement. The License Agreement is located at
/opt/nvidia/deepstream/deepstream/LicenseAgreement.pdf. If you do not agree
to the terms and conditions of the License Agreement do not use the software.

EOF
/opt/nvidia/nvidia_entrypoint.sh $@
