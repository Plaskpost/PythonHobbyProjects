import matplotlib.pyplot as plt

a = [0.07204654874674077, 0.08759481525139279, 0.10321992730786178, 0.11893582694220584, 0.13475665191458575, 0.15069676718795222, 0.16677079731714173, 0.18299365989773264, 0.19938060021878812, 0.21594722727286353, 0.23270955128570847, 0.24968402293794156, 0.2668875744628707, 0.2843376628178902, 0.3020523151411112, 0.32005017672142344, 0.33835056172799227, 0.3569735069664972, 0.37593982895049294, 0.3952711846024073, 0.4149901359269279, 0.43512021902927245, 0.4556860178869897, 0.47671324332203824, 0.49822881766218075, 0.520260965629987, 0.5428393120503898, 0.5659949870281196, 0.589760739312652, 0.614171058643791, 0.6392623079549082, 0.665072866404131, 0.6916432843105866, 0.7190164511911803, 0.7472377782265625, 0.776355396637058, 0.806420373618721, 0.8374869476813558, 0.8696127854514515, 0.9028592622451725, 0.9372917690020302, 0.9729800484859084, 1.0099985640247695, 1.0484269044757895, 1.0883502295766618, 1.1298597603876743, 1.173053320152249, 1.2180359316221399, 1.2649204777174958, 1.3138284333468846, 1.364890677315472, 1.4182483945261453, 1.474054080166539, 1.53247265930041, 1.5936827373059543, 1.6578779989586394, 1.7252687767319514, 1.796083812144957, 1.8705722378247458, 1.9490058124990242, 2.031681446511868, 2.118924061870416, 2.2110898384618167, 2.3085699072309183, 0.13544947848174616, -1.4210596611314372, -1.3688838548326885, -1.3186879318205058, -1.2703584131018886, -1.2237893192821332, -1.1788815663024366, -1.1355424164688657, -1.09368497902625, -1.053227755199714, -1.0140942231915915, -0.9762124591389352, -0.9395147904693033, -0.9039374784888423, -0.8694204273740667, -0.8359069170427347, -0.8033433576434987, -0.7716790636403914, -0.7408660456752472, -0.7108588185756446, -0.6816142240403877, -0.653091266678004, -0.625250962205854, -0.5980561967297149, -0.5714715961287311, -0.5454634046614757, -0.5199993719902665, -0.49504864789571457, -0.470581684018029, -0.44657014202063294, -0.4229868076250085, -0.39980551001252707, -80.78567383079726, 2.0514697999870393, 2.313070193310253, 2.6275923935654006, 3.0103471513568962, 3.4826135392012247, 4.074666849234447, 4.8307641833937325, 5.8176815155686725, 7.140021082487749, 8.96920937122178, 11.602229371204615, 15.590912167919484, 5.941915569001182, -0.08495121556826746, -0.06524426195397837, -0.04559429368715939, -0.025984101111802715, -0.0063965593109855945, 0.013185408430302914, 0.03277886697117083, 0.0524009056139505]
b = [0.056561350822548206, 0.07204654874674077, 0.08759481525139279, 0.10321992730786178, 0.11893582694220584, 0.13475665191458575, 0.15069676718795222, 0.16677079731714173, 0.18299365989773264, 0.19938060021878812, 0.21594722727286353, 0.23270955128570847, 0.24968402293794156, 0.2668875744628707, 0.2843376628178902, 0.3020523151411112, 0.32005017672142344, 0.33835056172799227, 0.3569735069664972, 0.37593982895049294, 0.3952711846024073, 0.4149901359269279, 0.43512021902927245, 0.4556860178869897, 0.47671324332203824, 0.49822881766218075, 0.520260965629987, 0.5428393120503898, 0.5659949870281196, 0.589760739312652, 0.614171058643791, 0.6392623079549082, 0.665072866404131, 0.6916432843105866, 0.7190164511911803, 0.7472377782265625, 0.776355396637058, 0.806420373618721, 0.8374869476813558, 0.8696127854514515, 0.9028592622451725, 0.9372917690020302, 0.9729800484859084, 1.0099985640247695, 1.0484269044757895, 1.0883502295766618, 1.1298597603876743, 1.173053320152249, 1.2180359316221399, 1.2649204777174958, 1.3138284333468846, 1.364890677315472, 1.4182483945261453, 1.474054080166539, 1.53247265930041, 1.5936827373059543, 1.6578779989586394, 1.7252687767319514, 1.796083812144957, 1.8705722378247458, 1.9490058124990242, 2.031681446511868, 2.118924061870416, 2.2110898384618167, 2.3085699072309183, 0.13544947848174616, -1.4210596611314372, -1.3688838548326885, -1.3186879318205058, -1.2703584131018886, -1.2237893192821332, -1.1788815663024366, -1.1355424164688657, -1.09368497902625, -1.053227755199714, -1.0140942231915915, -0.9762124591389352, -0.9395147904693033, -0.9039374784888423, -0.8694204273740667, -0.8359069170427347, -0.8033433576434987, -0.7716790636403914, -0.7408660456752472, -0.7108588185756446, -0.6816142240403877, -0.653091266678004, -0.625250962205854, -0.5980561967297149, -0.5714715961287311, -0.5454634046614757, -0.5199993719902665, -0.49504864789571457, -0.470581684018029, -0.44657014202063294, -0.4229868076250085, -0.39980551001252707, -80.78567383079726, 2.0514697999870393, 2.313070193310253, 2.6275923935654006, 3.0103471513568962, 3.4826135392012247, 4.074666849234447, 4.8307641833937325, 5.8176815155686725, 7.140021082487749, 8.96920937122178, 11.602229371204615, 15.590912167919484, 5.941915569001182, -0.08495121556826746, -0.06524426195397837, -0.04559429368715939, -0.025984101111802715, -0.0063965593109855945, 0.013185408430302914, 0.03277886697117083]
c = [0.015485197924192562, 0.01554826650465202, 0.015625112056468993, 0.015715899634344055, 0.01582082497237991, 0.015940115273366473, 0.01607403012918951, 0.016222862580590913, 0.016386940321055476, 0.01656662705407541, 0.016762324012844942, 0.016974471652233092, 0.01720355152492914, 0.01745008835501949, 0.017714652323221003, 0.017997861580312247, 0.01830038500656883, 0.01862294523850494, 0.018966321983995726, 0.01933135565191435, 0.019718951324520617, 0.020130083102344543, 0.020565798857717255, 0.021027225435048535, 0.021515574340142507, 0.02203214796780628, 0.022578346420402795, 0.023155674977729745, 0.023765752284532482, 0.024410319331138908, 0.02509124931111728, 0.025810558449222754, 0.026570417906455646, 0.027373166880593658, 0.02822132703538216, 0.029117618410495538, 0.030064976981662994, 0.0310665740626348, 0.0321258377700957, 0.03324647679372106, 0.034432506756857606, 0.035688279483878205, 0.03701851553886115, 0.03842834045101995, 0.039923325100872376, 0.041509530811012496, 0.04319355976457473, 0.0449826114698908, 0.04688454609535597, 0.048907955629388766, 0.051062243968587495, 0.05335771721067317, 0.055805685640393676, 0.05841857913387116, 0.06121007800554423, 0.06419526165268508, 0.06739077777331204, 0.07081503541300549, 0.07448842567978886, 0.07843357467427836, 0.08267563401284406, 0.08724261535854794, 0.09216577659140057, 0.09748006876910154, 2.173120428749172, 1.5565091396131834, 0.05217580629874874, 0.050195923012182675, 0.04832951871861724, 0.04656909381975538, 0.044907752979696625, 0.0433391498335709, 0.04185743744261572, 0.04045722382653594, 0.03913353200812253, 0.03788176405265631, 0.03669766866963187, 0.03557731198046099, 0.034517051114775654, 0.033513510331331986, 0.03256355939923594, 0.03166429400310733, 0.03081301796514424, 0.030007227099602574, 0.02924459453525685, 0.02852295736238375, 0.027840304472150024, 0.027194765476139082, 0.026584600600983777, 0.026008191467255415, 0.025464032671209225, 0.0249507240945519, 0.024466963877685544, 0.02401154199739608, 0.023583334395624433, 0.023181297612481444, 80.38586832078474, 82.83714363078431, 0.2616003933232136, 0.31452220025514777, 0.3827547577914956, 0.4722663878443285, 0.5920533100332221, 0.7560973341592856, 0.98691733217494, 1.3223395669190765, 1.8291882887340307, 2.633019999982835, 3.988682796714869, 9.648996598918302, 6.0268667845694495, 0.01970695361428909, 0.019649968266818973, 0.019610192575356677, 0.01958754180081712, 0.01958196774128851, 0.019593458540867914, 0.01962203864277967]
d = [88.40155780789544, 88.48915262314684, 88.5923725504547, 88.7113083773969, 88.84606502931149, 88.99676179649944, 89.16353259381658, 89.34652625371432, 89.5459068539331, 89.76185408120597, 89.99456363249168, 90.24424765542962, 90.51113522989249, 90.79547289271038, 91.09752520785149, 91.41757538457291, 91.7559259463009, 92.1128994532674, 92.4888392822179, 92.8841104668203, 93.29910060274723, 93.7342208217765, 94.1899068396635, 94.66662008298553, 95.16484890064771, 95.6851098662777, 96.22794917832809, 96.79394416535621, 97.38370490466886, 97.99787596331265, 98.63713827126756, 99.30221113767169, 99.99385442198228, 100.71287087317346, 101.46010865140002, 102.23646404803708, 103.0428844216558, 103.88037136933715, 104.7499841547886, 105.65284341703378, 106.59013518603581, 107.56311523452172, 108.57311379854649, 109.62154070302228, 110.70989093259894, 111.83975069298661, 113.01280401313886, 114.230839944761, 115.4957604224785, 116.80958885582538, 118.17447953314085, 119.592727927667, 121.06678200783354, 122.59925466713395, 124.1929374044399, 125.85081540339854, 127.5760841801305, 129.37216799227545, 131.2427402301002, 133.19174604259922, 135.2234274891111, 137.3423515509815, 139.55344138944332, 141.86201129667424, 141.997460775156, 140.57640111402455, 139.20751725919186, 137.88882932737135, 136.61847091426947, 135.39468159498733, 134.2158000286849, 133.08025761221603, 131.98657263318978, 130.93334487799007, 129.91925065479847, 128.94303819565954, 128.00352340519024, 127.0995859267014, 126.23016549932733, 125.39425858228459, 124.5909152246411, 123.8192361610007, 123.07837011532546, 122.36751129674981, 121.68589707270942, 121.03280580603142, 120.40755484382557, 119.80949864709585, 119.23802705096712, 118.69256364630564, 118.17256427431538, 117.67751562641966, 117.20693394240163, 116.760363800381, 116.33737699275599, 115.93757148274346, 35.1518976519462, 37.20336745193324, 39.51643764524349, 42.14403003880889, 45.15437719016579, 48.636990729367014, 52.71165757860146, 57.54242176199519, 63.360103277563866, 70.50012436005161, 79.4693337312734, 91.07156310247801, 106.6624752703975, 112.60439083939868, 112.51943962383041, 112.45419536187643, 112.40860106818927, 112.38261696707747, 112.37622040776648, 112.38940581619678, 112.42218468316796, 112.4745855887819]

plt.scatter(range(len(c)), a, label="a")
plt.scatter(range(len(c)), b, label="b")
plt.scatter(range(len(c)), c, label="c")
plt.scatter(range(len(c)), d, label="d")
plt.legend()
plt.show()