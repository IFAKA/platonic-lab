| Method | Setting | Seeds | mutual 10-NN (mean, range) | image->caption R@1 |
|---|---|---|---|---|
| good | same speed (5e-4), lr 0.0005 | 3 | 0.320 (0.314-0.325) | 0.084 |
| good | own best speed, lr 0.00125 | 1 | 0.328 (0.328-0.328) | 0.082 |
| bad | same speed (5e-4), lr 0.0005 | 3 | 0.192 (0.187-0.197) | 0.063 |
| bad | own best speed, lr 0.00125 | 1 | 0.196 (0.196-0.196) | 0.068 |
| ntp | same speed (5e-4), lr 0.0005 | 3 | 0.148 (0.130-0.164) | 0.057 |
| ntp | own best speed, lr 0.003 | 1 | 0.168 (0.168-0.168) | 0.049 |

Chance: mutual 10-NN 0.010, R@1 0.001. Test set: 1000 held-out photos.
