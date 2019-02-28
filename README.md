# BareNet
Open Sourcing Unbiased Nudity Detection

# Note: Entire credit for collecting this dataset goes to Jae Jin https://github.com/Kadantte

![](https://i.imgur.com/5J9ESnu.png) ![](https://i.imgur.com/Fs6exOx.png)


# Classes
X_BELLY -> Exposed Belly

X_BUTTOCKS -> Exposed Buttocks

X_BREAST -> Exposed Female Breast

C_BREAST -> Covered Female Breast

X_FE_GENITALIA -> Exposed Female Genitalia

C_FE_GENITALIA -> Covered Female Genitalia

X_M_GENETALIA -> Exposed Male Genetalia

X_M_BREAST -> Exposed Male Breast

C_M_GENETALIA -> Covered Male Genetalia (Coming Soon)

# Insallation
```
pip install barenet
```

# Usage
```
from barenet import BareNet
detector = BareNet('checkpoint_path')
detector.detect('bikini_girl.jpg', thresh=0.4)
# [{'class': 'C_BREAST', 'prob': 0.983704, 'box': [251, 388, 411, 564]}, {'class': 'C_BREAST', 'prob': 0.98258233, 'box': [435, 408, 570, 570]}, {'class': 'X_BELLY', 'prob': 0.9819952, 'box': [312, 566, 545, 828]}, {'class': 'C_FE_GENITALIA', 'prob': 0.8241659, 'box': [398, 877, 518, 992]}]
```
