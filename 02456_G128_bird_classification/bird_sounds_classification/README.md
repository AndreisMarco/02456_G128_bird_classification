---
license: apache-2.0
metrics:
- accuracy
- f1
base_model:
- facebook/wav2vec2-base-960h
---
See https://www.kaggle.com/code/dima806/bird-species-by-sound-detection for more details.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6449300e3adf50d864095b90/apB76WWSjVeQEevcxekNg.png)

```
Classification report:

                            precision    recall  f1-score   support

               Andean Guan     0.9310    0.8182    0.8710        33
            Andean Tinamou     0.5000    0.7083    0.5862        24
    Australian Brushturkey     0.7500    0.1765    0.2857        17
          Band-tailed Guan     0.7436    0.7838    0.7632        37
         Bartletts Tinamou     0.9579    0.9891    0.9733        92
              Bearded Guan     0.8889    0.8649    0.8767        37
      Black-capped Tinamou     0.4154    0.9000    0.5684        30
 Blue-throated Piping Guan     0.0000    0.0000    0.0000        22
         Brazilian Tinamou     0.8026    0.8841    0.8414        69
             Brown Tinamou     0.9391    0.9076    0.9231       119
         Brushland Tinamou     0.9048    0.8636    0.8837        22
                Cauca Guan     0.9925    0.9778    0.9851       135
          Chaco Chachalaca     0.9383    1.0000    0.9682        76
Chestnut-winged Chachalaca     0.8108    0.8108    0.8108        37
         Cinereous Tinamou     0.9737    0.9867    0.9801        75
      Colombian Chachalaca     1.0000    0.8649    0.9275        37
              Crested Guan     0.9574    0.9375    0.9474        48
            Dusky Megapode     0.7143    0.9722    0.8235        36
         Dusky-legged Guan     0.8857    0.9394    0.9118        33
             Dwarf Tinamou     0.8750    0.5600    0.6829        25
             Great Tinamou     0.9697    0.9846    0.9771        65
              Grey Tinamou     0.9492    0.9655    0.9573        58
    Grey-headed Chachalaca     0.8667    0.9123    0.8889        57
          Highland Tinamou     1.0000    0.9273    0.9623        55
         Little Chachalaca     0.7632    0.9355    0.8406        31
            Little Tinamou     0.7419    0.8519    0.7931        27
   Orange-footed Scrubfowl     0.9640    0.9640    0.9640       111
       Pale-browed Tinamou     0.6667    0.0909    0.1600        22
          Plain Chachalaca     0.9390    0.9390    0.9390        82
        Red-legged Tinamou     0.7297    0.9310    0.8182        29
        Red-winged Tinamou     0.8605    0.9487    0.9024        39
 Rufous-bellied Chachalaca     0.9911    0.9407    0.9652       118
  Rufous-headed Chachalaca     0.8333    0.7143    0.7692        28
  Rufous-vented Chachalaca     0.8478    0.8667    0.8571        45
       Rusty-margined Guan     0.8889    0.9412    0.9143        34
    Slaty-breasted Tinamou     0.8649    0.9143    0.8889        35
      Small-billed Tinamou     0.7742    0.8889    0.8276        27
          Solitary Tinamou     0.6786    0.6786    0.6786        28
       Speckled Chachalaca     0.9333    0.9655    0.9492        58
                Spixs Guan     0.9600    0.7500    0.8421        32
           Spotted Nothura     0.7234    0.9714    0.8293        35
           Tataupa Tinamou     0.6571    0.7931    0.7188        29
    Tawny-breasted Tinamou     0.9138    0.9138    0.9138        58
           Thicket Tinamou     0.9663    0.9773    0.9718        88
         Undulated Tinamou     0.9315    0.8095    0.8662        84
        Variegated Tinamou     1.0000    0.2105    0.3478        19
   West Mexican Chachalaca     0.8615    0.9655    0.9106        58
     White-bellied Nothura     0.8000    0.7273    0.7619        22
    White-throated Tinamou     0.0000    0.0000    0.0000        14
     Yellow-legged Tinamou     0.9623    0.9808    0.9714        52

                  accuracy                         0.8822      2444
                 macro avg     0.8204    0.8081    0.7959      2444
              weighted avg     0.8806    0.8822    0.8727      2444
```