import numba
from numba import njit
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from scipy import stats

COD = np.array([[1.19793333],[1.31593295],[2.34966772]]) # COD for sugar,proteins,lipids
xi_frac, xt_frac = (0.1320097462686567, 0.24711158955223883)
fw_data = np.array([[8.88414082e+01, 9.05808079e+01, 8.14248902e+01, 6.45147617e+01,
        8.79827498e+01, 5.30803977e+01, 5.24609732e+01, 0.00000000e+00,
        9.14690277e+01, 9.68944364e+01, 9.13350378e+01, 9.61096075e+01,
        7.56488483e+01, 6.14364095e+01, 5.65429464e+01, 0.00000000e+00,
        0.00000000e+00, 7.53865398e+01, 7.91758379e+01, 7.54185143e+01,
        8.94281336e+01, 4.80041171e+01, 8.03853608e+01, 6.46862523e+01,
        9.18041206e+01, 2.14141122e+01, 6.49839966e+01, 4.78883341e+01,
        9.63076120e+01, 6.35067810e+01, 4.65950809e+01, 5.45207857e+00,
        3.58538692e+01, 8.97427183e+01, 7.71787088e+01, 5.82953387e+01,
        5.37128110e+01, 6.35848800e+01, 8.96249868e+01, 1.13390907e+01,
        9.11155877e+01, 7.94940022e+01, 9.22224187e+01, 1.54984339e+01,
        8.04293930e+01, 7.33416837e+01, 7.10062591e+01, 5.06633879e+01,
        5.66742213e+01, 3.45895811e+01, 6.28437945e+01, 9.36840830e+01,
        9.44056870e+01, 9.56087065e+01, 8.89737706e+01, 1.14873024e+01,
        6.85433276e+01, 9.99873638e+01, 9.99327708e+01, 9.05125377e+01,
        0.00000000e+00, 9.99998685e+01, 8.30023254e+01, 9.99873638e+01,
        9.99873638e+01, 7.59795628e+01, 6.55028493e+01, 6.29147768e+01,
        1.90311934e+01, 7.66881779e+01, 1.03850080e+00, 2.95513299e+01,
        1.44378781e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.80158078e+01, 4.62572903e+01, 6.06462772e+01,
        5.44756057e+01, 8.74413262e+01, 2.84545777e+01, 6.56097686e+01,
        9.99873638e+01, 9.99970491e+01, 9.99873638e+01, 1.89231587e+01,
        3.31743777e+01, 3.51890086e+01, 2.51925201e+01, 4.23582928e+01,
        3.79703598e+01, 9.88131343e+01, 4.40415202e+01],
       [7.38785164e+00, 9.37264860e+00, 5.50145150e+00, 4.70699427e+00,
        1.19972076e+01, 5.34367506e+00, 2.83298256e+01, 7.28510470e+01,
        8.52669045e+00, 3.10223175e+00, 8.66104329e+00, 3.88696771e+00,
        1.55671579e+01, 2.41519956e+01, 1.19721654e+01, 3.92900816e+01,
        0.00000000e+00, 1.01099342e+01, 1.65644626e+01, 1.71812130e+01,
        1.05678409e+01, 1.76919164e+01, 1.49724000e+01, 2.59928012e+01,
        8.16412378e+00, 5.48033929e+00, 6.60060001e+00, 1.18109568e+01,
        3.61348729e+00, 7.17458185e+00, 4.01545792e+00, 6.46131100e+00,
        5.86548033e+00, 5.49627098e+00, 5.36889481e+00, 4.55091336e+00,
        5.91698749e+00, 4.63286890e+00, 9.63784266e+00, 1.09760940e+01,
        8.87963469e+00, 2.04959779e+01, 7.74178113e+00, 5.02909274e+01,
        1.95258166e+01, 8.25345561e+00, 9.21915457e+00, 1.03905384e+01,
        1.09345175e+01, 1.01135280e+01, 9.92797186e+00, 6.29314243e+00,
        5.56488106e+00, 4.36387836e+00, 4.61984815e+00, 9.91915613e+00,
        5.75595784e+00, 0.00000000e+00, 0.00000000e+00, 8.68406598e+00,
        3.72853781e+01, 0.00000000e+00, 1.62932457e+01, 0.00000000e+00,
        0.00000000e+00, 2.09024879e+01, 2.77337026e+01, 2.93086522e+01,
        5.46252209e+01, 2.30071227e+01, 3.28251210e+01, 2.82612850e+01,
        3.25761927e+01, 2.00939302e-01, 1.80197498e+01, 0.00000000e+00,
        0.00000000e+00, 2.28681777e+01, 1.88366004e+01, 3.06374591e+01,
        1.62856012e+01, 5.39291988e+00, 1.98263483e+01, 2.76632918e+01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.65167167e+01,
        2.63897126e+00, 3.29059975e+00, 1.12897421e+00, 4.91395967e+00,
        3.28200709e+00, 2.65861557e-01, 5.67580638e-01],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.07526204e+01,
        0.00000000e+00, 8.10466143e+00, 8.77560321e-01, 2.71313091e+01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        8.78190535e+00, 1.44095308e+01, 3.14808490e+01, 6.06121731e+01,
        9.90819000e+01, 1.45004360e+01, 4.25744099e+00, 7.39812183e+00,
        0.00000000e+00, 3.43004617e+01, 4.64003169e+00, 9.31837299e+00,
        0.00000000e+00, 1.03232284e+01, 0.00000000e+00, 2.36258821e+01,
        7.41815572e-02, 3.84816482e+00, 0.00000000e+00, 5.96262153e+01,
        1.61374591e+01, 4.75805534e+00, 4.05027283e+00, 0.00000000e+00,
        4.03593967e+01, 1.39996206e-07, 7.26264380e-01, 3.11552942e+01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.41883890e+01,
        0.00000000e+00, 1.84009893e+01, 1.97729910e+01, 0.00000000e+00,
        0.00000000e+00, 7.99905482e-08, 2.72267668e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 6.38053942e+00, 4.71718596e+00,
        3.97729414e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 7.02042102e-01, 0.00000000e+00,
        0.00000000e+00, 3.11553168e+00, 6.76010739e+00, 7.47800989e+00,
        1.77380626e+01, 0.00000000e+00, 6.61319264e+01, 4.21847119e+01,
        5.29830984e+01, 9.97140193e+01, 8.17413207e+01, 9.96188987e+01,
        9.96492055e+01, 2.91135804e+01, 3.49027172e+01, 8.01700717e+00,
        2.92354593e+01, 7.16256937e+00, 5.17140334e+01, 6.72361177e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.45560118e+01,
        4.99556985e+01, 6.15066563e+01, 5.31572581e+01, 5.27128571e+01,
        5.87396049e+01, 0.00000000e+00, 5.53844678e+01]])
aw_data = np.array([[6.53818948e+01, 6.38992157e+01, 7.72092429e+01, 8.21755888e+01,
        6.78422913e+01, 8.12361589e+01, 8.14951473e+01, 8.64146261e+01,
        9.65422456e+01, 9.89364859e+01, 7.86029622e+01, 4.59098119e+01,
        7.07174332e+01, 9.76965714e+01, 7.51635856e+01, 9.82847384e+01,
        7.56443049e+01, 9.99996657e+01, 8.08957161e+01, 9.96068318e+01,
        9.99996180e+01, 9.80752057e+01, 9.99586354e+01, 9.99993639e+01,
        9.99994393e+01, 9.92611377e+01, 9.99997073e+01, 9.99996243e+01,
        9.99998002e+01, 9.99988887e+01, 7.99808200e+01, 9.88563458e+01,
        9.01778171e+01, 9.70865004e+01, 8.74181738e+01, 9.99996936e+01,
        9.99995781e+01, 9.95308143e+01, 6.91057075e+01, 8.25421779e+01,
        7.95842287e+01, 1.48951580e+01, 5.52296428e+01, 8.41153018e+01,
        9.05719555e+01, 4.55701346e+01, 9.64694365e+01, 8.98750481e+01,
        9.49476931e+01, 9.74242050e+01, 9.65987677e+01, 3.45565452e+01,
        9.44172965e+01, 4.37599306e+01, 4.81140086e+01, 9.51682490e+01,
        7.94270122e+01, 8.93820778e+01, 8.80776813e+01, 6.14867305e+01,
        7.56742830e+01, 7.12622170e+01, 8.76127487e+01, 8.91526477e+01,
        8.88631310e+01, 8.84118612e+01, 8.79611989e+01, 6.40461702e+01,
        7.90606078e+01, 3.54526578e+01, 4.15426274e+01, 3.48075590e+01,
        3.36573741e+01, 4.40261590e+01, 5.54851016e+01, 6.42979236e+01,
        9.99996044e+01, 4.11770451e+01, 5.80553378e+01, 6.66456257e+01,
        5.77421954e+01, 6.39767212e+01, 6.12725184e+01, 7.46852541e+01,
        7.23365260e+01, 7.03756612e+01, 7.16009349e+01, 7.83567338e+01,
        5.87289072e+01, 7.62604966e+01, 5.20620318e+01, 2.78155826e+01,
        8.71083464e+01, 3.86415904e+01, 2.07889840e+01, 1.74797701e+01,
        2.87282916e+00, 2.27994934e-06, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 9.20007550e+01, 3.36097851e+01, 1.30386862e+00,
        3.53770230e+01, 2.84283897e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.56042264e+01, 6.79470456e+01, 6.48838013e+01,
        6.44851128e+01, 0.00000000e+00],
       [1.24424536e+01, 1.12586487e+01, 3.24231275e+00, 6.27245412e+00,
        9.46220115e+00, 6.53678101e+00, 1.54152161e+00, 3.12885778e+00,
        3.43558506e+00, 1.05324065e+00, 2.05426971e+00, 2.28471478e+00,
        2.32994426e+00, 2.29028411e+00, 4.34260940e+00, 1.70356533e+00,
        4.34654223e+00, 0.00000000e+00, 1.90844377e+01, 2.63714281e-01,
        0.00000000e+00, 2.69481107e-01, 4.12557626e-02, 0.00000000e+00,
        0.00000000e+00, 2.63720428e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 4.57453127e+00, 1.12825424e+00,
        2.72467929e+00, 2.88426187e+00, 9.44289935e+00, 0.00000000e+00,
        0.00000000e+00, 2.63717004e-01, 2.13280538e+01, 1.09021771e+01,
        1.34342656e+01, 3.76473722e+00, 1.49147405e+01, 1.10642439e+01,
        3.83545215e+00, 3.49021183e+00, 3.52255086e+00, 6.31053617e+00,
        5.04731422e+00, 0.00000000e+00, 2.40826749e+00, 2.43844807e+00,
        4.59559761e+00, 3.87454080e+00, 5.09586659e+00, 4.79676382e+00,
        1.38091360e+01, 1.06045467e+01, 9.88037163e+00, 1.34106642e+01,
        1.04202318e+01, 1.32510079e+01, 1.22400953e+01, 1.08414024e+01,
        1.11257196e+01, 1.15723454e+01, 1.20217123e+01, 2.51757500e+01,
        1.73785228e+01, 5.66255333e+00, 1.96708865e+01, 6.01881695e+00,
        3.86902748e+01, 7.52361841e+00, 1.43489639e+01, 1.19774813e+01,
        0.00000000e+00, 3.59022150e+01, 4.01103948e+01, 1.98364098e+01,
        2.40352925e+01, 2.17400061e+01, 2.16961092e+01, 2.52757823e+01,
        2.76408366e+01, 2.96021928e+01, 2.83887738e+01, 1.84314735e+01,
        2.39216111e+01, 2.07183476e+01, 3.03029631e+01, 4.59671852e+01,
        1.14603809e+01, 1.85850545e+01, 1.89328648e+01, 1.76837936e+01,
        1.89729668e+01, 2.03411792e+01, 2.14751018e+01, 2.29633337e+01,
        2.31108367e+01, 7.94590154e+00, 2.99671628e+01, 2.85700498e+01,
        5.64225894e+00, 1.23958066e+01, 2.36816023e+01, 2.80139015e+01,
        2.10709131e+01, 5.43703196e+01, 2.31273905e+01, 2.70980802e+01,
        3.55083564e+01, 1.92810509e+01],
       [1.25808536e+01, 8.47858525e+00, 1.57591350e+01, 1.15474014e+01,
        2.26944030e+01, 1.22160070e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.53411311e+01, 7.99947711e-08,
        0.00000000e+00, 0.00000000e+00, 9.99979741e-08, 0.00000000e+00,
        1.99999469e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 3.13664731e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.56406018e+00, 6.55330290e+00,
        6.97939595e+00, 8.13302339e+01, 2.98443190e+01, 3.87063439e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 2.57462665e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.50626774e+01,
        1.38565614e+01, 1.54465709e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.07752266e+01,
        3.55810723e+00, 5.59453771e+00, 4.34495950e+00, 5.91626488e+01,
        2.76223103e+01, 1.75090772e+01, 0.00000000e+00, 1.19996626e-06,
        0.00000000e+00, 2.29035600e+01, 1.82663908e+00, 1.35094375e+01,
        1.82176175e+01, 7.56511556e+00, 1.70195517e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.20962716e+00,
        9.69275711e+00, 3.01874054e+00, 1.76326643e+01, 2.21627900e+01,
        1.42901967e+00, 7.99952382e-08, 9.99922527e-08, 5.61204388e+01,
        2.33702194e+01, 3.05974505e+01, 7.84567412e+01, 7.69775655e+01,
        7.68274300e+01, 0.00000000e+00, 2.37941492e+01, 3.90279298e+01,
        5.48681008e+00, 1.37461783e+01, 1.53184999e+00, 7.19536325e+01,
        7.88793084e+01, 0.00000000e+00, 8.92232447e+00, 8.01458649e+00,
        0.00000000e+00, 8.06803927e+01]])
mw_data = np.array([[9.99985435e+01, 2.04072290e+00, 9.39688588e+01, 3.73890310e+01,
        5.57412021e+01, 2.39981633e-07, 3.52080700e+01, 5.37593221e+01,
        3.88550598e+01, 5.66237671e+01, 6.51115098e+01, 6.10905709e+00,
        3.95928852e+01, 1.31065114e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.10845278e+01, 1.49982845e+01, 3.39982123e-07,
        1.69996430e-06, 7.99944439e-08, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.43559087e+01, 5.99845007e-08, 2.93165978e+00,
        5.99861341e-08, 5.99856340e-08, 0.00000000e+00, 7.99649139e-08,
        0.00000000e+00, 4.63065158e+01, 9.36439379e+01, 9.79443363e+01,
        5.58205287e+01, 5.07425059e+01, 5.14441235e+01, 2.24942838e+01,
        7.03873793e+01, 4.39998935e+01, 8.53179912e+01, 7.87304584e+01,
        5.30933052e+01, 8.59997550e+01, 9.99998123e+01, 9.99994797e+01,
        9.99976625e+01, 9.99972297e+01, 9.99971237e+01, 9.99981141e+01,
        9.99971516e+01, 9.99979608e+01, 9.99970491e+01, 9.99984285e+01,
        9.99991315e+01, 9.99981245e+01, 9.99970491e+01],
       [0.00000000e+00, 2.05240458e+01, 5.98415589e+00, 1.43169308e+01,
        1.25943014e+01, 1.00428621e+01, 2.07927445e+01, 1.14478785e+01,
        2.16454386e+01, 1.65434544e+01, 1.26571251e+01, 9.30335452e+00,
        1.15846664e+01, 1.12527439e+01, 9.75438387e+00, 1.22671795e+01,
        1.31018355e+01, 3.21403772e+01, 2.77483889e+01, 1.91407166e+01,
        2.13489684e+01, 2.55357039e+01, 2.13829685e+01, 2.25746214e+01,
        2.54330911e+01, 1.74266713e+01, 1.98101344e+01, 1.20129562e+01,
        1.45257437e+01, 1.43870817e+01, 1.52356403e+01, 1.67456716e+01,
        1.82228731e+01, 1.36880054e+01, 6.07456883e+00, 1.98480573e+00,
        9.84559836e-01, 1.82257300e+01, 1.04389484e+01, 1.18621974e+01,
        1.70319424e+01, 3.67458084e+01, 7.37971428e+00, 7.01138087e+00,
        1.40463407e+01, 1.39901533e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [0.00000000e+00, 7.74269356e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.08017996e+01, 4.39955028e+01, 2.31405315e+01,
        3.94967658e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.52032613e-01,
        9.81682968e+00, 5.99941502e-08, 4.90514655e+01, 5.22540242e+01,
        4.29143872e+01, 5.80179871e+01, 7.85868034e+01, 7.35963984e+01,
        7.45276670e+01, 0.00000000e+00, 2.28666431e+01, 5.99956308e-08,
        2.53569483e+01, 2.53888271e+01, 5.53303868e+01, 4.95806143e+01,
        8.16921849e+01, 2.99276313e+00, 0.00000000e+00, 0.00000000e+00,
        7.99977425e-08, 3.10201221e+01, 3.81084303e+01, 5.89549861e+01,
        9.06726145e+00, 1.92462625e+01, 7.29492218e+00, 1.42504591e+01,
        3.28528989e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

fw_label = np.array(['veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'meat', 'grain',
       'veg', 'veg', 'veg', 'veg', 'grain', 'other', 'other', 'other',
       'veg', 'other', 'other', 'veg', 'veg', 'veg', 'other', 'veg',
       'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg',
       'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'grain',
       'meat', 'veg', 'other', 'other', 'other', 'other', 'other', 'food',
       'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'veg',
       'veg', 'veg', 'veg', 'veg', 'veg', 'veg', 'food', 'food', 'food',
       'food', 'food', 'food', 'other', 'other', 'other', 'food', 'food',
       'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food',
       'food', 'food', 'food', 'other', 'other', 'other', 'other', 'food',
       'food', 'meat'])
mw_label = np.array(['msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw',
       'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw',
       'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw',
       'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw',
       'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw',
       'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'msw', 'textile',
       'textile', 'textile', 'wood', 'msw', 'msw', 'paper'])

label_dict = {'food':1, 'veg':0.492, 'grain':0.081, 'meat':0.173, 'other':0.148, 
              'msw':1, 'textile':0.0583, 'wood':0.0619, 'paper':0.2305}
fw_weight = np.array([label_dict[i] for i in fw_label])
aw_weight = np.ones(len(aw_data[0]))
mw_weight = np.array([label_dict[i] for i in mw_label])

def rand_data(kernel, n=1000, threshold=1e-5):
    l, h = 0, 100
    x = (h - l)*np.random.rand(n) + l
    y = (h - l)*np.random.rand(n) + l
    z = (h - l)*np.random.rand(n) + l
    density = kernel(np.array([x,y,z]))
    filter = density > threshold
    density = density[filter]
    x = x[filter]
    y = y[filter]
    z = z[filter]
    return x,y,z,density

def resample(kernel, n=1):
    xyz = np.clip(kernel.resample(n), a_min=10, a_max=90) * COD
    xyz *= .01
    return xyz.T

def create_rand_dist(kernels, n=10):
    l = len(kernels)
    out = np.zeros([n,l,3]) 
    for i in range(l):
        out[:,i,:] = resample(kernels[i], n)
    out = np.nan_to_num(out, nan=1)
    return out

@numba.cfunc(lsoda_sig)
def Model_lsoda(t, Z, dz, p):
  ## 1. Input parameters

    Alpha1, Alpha2, Alpha3, Q, Beta = p[0], p[1], p[2], p[3], p[4]
    V  = 3400. # volume of reactor
    Vg = 300.  # gas volume of reactor
    Xi = Q * 0.247 * Beta

  ## 2. Constants
    # Thermal Constants
    R    = 0.083145 #bar.M^-1.K^-1
    Patm = 1.013 #bar
    T0   = 298.15 #K
    T    = 308.15  #K

    # Stoichiometric Constants
    # Inhibitor
    Nxc  = 0.0376 / 14 # [-]
    NI   = 0.06 / 14   # [kmolN kg-1 COD]
    Naa  = 0.007       # [kmolN kg-1 COD]
    Nbac = 0.08 / 14   # [kmolN kg-1 COD]
    # Inhibitor and composites
    fsi_xc = 0.1  # Insoluble inert inhibitor stoichiometric
    fxi_xc = 0.2  # Soluble inert inhibitor stoichiometric
    fch_xc = 0.2  # carbohydrates stoichiometric 
    fli_xc = 0.3  # lipids stoichiometric
    fpr_xc = 0.2  # proteins stoichiometric
    # liquid
    ffa_li = 0.95
    fh2_su = 0.19
    fbu_su = 0.13
    fpro_su= 0.27
    fac_su = 0.41
    fh2_aa = 0.06
    fva_aa = 0.23
    fbu_aa = 0.26
    fpro_aa= 0.05
    fac_aa = 0.40 
    Ysu    = 0.1
    Yaa    = 0.08
    Yfa    = 0.06
    Yc4    = 0.06
    Ypro   = 0.04
    Yac    = 0.05
    Yh2    = 0.06

    # Rate constants
    kdis  = 0.5  # Disintegration rate constant [d-1] 
    kh_ch = 10   # Hydrolysis of carbohydrates rate constant [d-1] 
    kh_pr = 10   # Hydrolysis of proteins rate constant [d-1] 
    kh_li = 10   # Hydrolysis of lipids rate constant [d-1] 
    # Liquid Phase
    k_msu  = 30   # [d-1]
    K_Ssu  = 0.5  # [kgCOD m-3]
    k_maa  = 50   # [d-1]
    K_Saa  = 0.3  # [kgCOD m-3]
    k_mfa  = 6    # [d-1]
    K_Sfa  = 0.4  # [kgCOD m-3]
    k_mc4  = 20   # [d-1]
    K_Sc4  = 0.2  # [kgCOD m-3]
    k_mpro = 13   # [d-1]
    K_Spro = 0.1  # [kgCOD m-3]
    k_mac  = 8    # [d-1]
    K_Sac  = 0.15 # [kgCOD m-3]
    k_mh2  = 35   # [d-1]
    K_Sh2  = 7e-6 # [kgCOD m-3]
    kdec_Xsu = 0.02  # [d-1]
    kdec_Xaa = 0.02  # [d-1]
    kdec_Xfa = 0.02  # [d-1]
    kdec_Xc4 = 0.02  # [d-1]
    kdec_Xpro= 0.02  # [d-1]
    kdec_Xac = 0.02  # [d-1]
    kdec_Xh2 = 0.02  # [d-1]
    # Inhibitor
    KS_IN    = 1e-4   # [M]
    K_Ih2fa  = 5e-6   # [kgCOD m-3]
    K_Ih2c4  = 1e-5   # [kgCOD m-3]
    K_Ih2pro = 3.5e-6 # [kgCOD m-3]
    K_Inh3   = 0.0018 # [M]
    # rate constants
    ph2o  = 0.0557
    kp    = 5e4      # [m3.d-1.bar-1] overhead pressure 
    KLa   = 200      # [d-1]
    KHco2 = 0.027147 # [Mliq.bar-1]
    KHch4 = 0.001162 # [Mliq.bar-1]
    KHh2  = 7.38e-4  # [Mliq.bar-1]

    # pH and ion
    pH_UL_aa =  5.5
    pH_LL_aa =  4
    pH_UL_ac =  7
    pH_LL_ac =  6
    pH_UL_h2 =  6
    pH_LL_h2 =  5
    K_w = 2.08e-14
    K_a_va  = 1.38e-5
    K_a_bu  = 1.5e-5
    K_a_pro = 1.32e-5
    K_a_ac  = 1.74e-5
    K_a_co2 =  4.94e-7
    K_a_IN  =  1.11e-9
    k_A_B_va  =  10e10 # M-1 d-1
    k_A_B_bu  =  10e10 # M-1 d-1
    k_A_B_pro =  10e10 # M-1 d-1
    k_A_B_ac  =  10e10 # M-1 d-1
    k_A_B_co2 =  10e10 # M-1 d-1
    k_A_B_IN  =  10e10 # M-1 d-1

    # Carbon balance
    C_xc  =  0.02786 # [kmole C.kg^-1COD]
    C_sI  =  0.03    # [kmole C.kg^-1COD]
    C_ch  =  0.0313  # [kmole C.kg^-1COD]
    C_pr  =  0.03    # [kmole C.kg^-1COD]
    C_li  =  0.022   # [kmole C.kg^-1COD]
    C_xI  =  0.03    # [kmole C.kg^-1COD]
    C_su  =  0.0313  # [kmole C.kg^-1COD]
    C_aa  =  0.03    # [kmole C.kg^-1COD]
    C_fa  =  0.0217  # [kmole C.kg^-1COD]
    C_bu  =  0.025   # [kmole C.kg^-1COD]
    C_pro =  0.0268  # [kmole C.kg^-1COD]
    C_ac  =  0.0313  # [kmole C.kg^-1COD]
    C_bac =  0.0313  # [kmole C.kg^-1COD]
    C_ch4 =  0.0156  # [kmole C.kg^-1COD]
    C_va  =  0.024   # [kmole C.kg^-1COD]
    
  ## 3. Reactor states
    Xi_ch = Alpha1 * Xi  # input of carbohydrates
    Xi_li = Alpha2 * Xi  # input of lipids
    Xi_pr = Alpha3 * Xi  # input of proteins

    # Particulate Matter
    X    = Z[0]  # 13 composite concentration
    X_ch = Z[1]  # 14 carbohydrates concentration
    X_li = Z[2]  # 16 lipids concentration
    X_pr = Z[3]  # 15 proteins concentration

    # # Liquid Phase
    S_su = Z[4]  # 1 monosaccharides concentration [kgCOD m-3]
    S_aa = Z[5]  # 2 amino acids concentration [kgCOD m-3]
    S_fa = Z[6]  # 3 long chain fatty acids concentration [kgCOD m-3]
    S_va = Z[7]  # 4 Total valerate concentration [kgCOD m-3]
    S_bu = Z[8]  # 5 Total butyrate concentration [kgCOD m-3]
    S_pro= Z[9]  # 6 Total propionate concentration [kgCOD m-3]
    S_ac = Z[10] # 7 Total acetate concentration [kgCOD m-3]
    S_h2 = Z[11] # 8 Total hydrogen gas concentration in liquid [kgCOD m-3]
    S_ch4= Z[12] # 9 Total methane gas concentration in liquid [kgCOD m-3]
    S_IN = Z[13] #10 Total Inhibitory nitorgen in system
    
    # Gas States
    G_h2  = Z[14]
    G_ch4 = Z[15]
    
    # Particulate Matter Continued
    X_su = Z[16]  # 17 sugars concentration
    X_aa = Z[17]  # 18 amino acids concentration
    X_fa = Z[18]  # 19 fatty acids concentration
    X_c4 = Z[19]  # 20 valerate concentration
    X_pro= Z[20]  # 21 propionate concentration
    X_ac = Z[21]  # 22 acetate concentration
    X_h2 = Z[22]  # 23 hydrogen concentration
    G_co2= Z[23] 
    S_IC = Z[24]
    S_cat_i = Z[25]
    S_an_i = Z[26]
    S_va_i = Z[27]
    S_bu_i = Z[28]
    S_pro_i = Z[29]
    S_ac_i = Z[30]
    S_hco3_i = Z[31]
    S_nh3 = Z[32]
  
  ## 4. Reaction rates
    # DAE 
    S_nh4_i = S_IN - S_nh3; 
    S_co2 = S_IC - S_hco3_i; 
    phi = S_cat_i + S_nh4_i - S_hco3_i - S_ac_i/64 - S_pro_i/112 - S_bu_i/160 - S_va_i/208 - S_an_i; 
    S_H_i = -phi*0.5 + 0.5*(phi**2+4*K_w)**0.5; 
    
    # Gas phase
    ph2  = G_h2  * R * T / 16
    pch4 = G_ch4 * R * T / 64
    pco2 = G_co2 * R * T
    Pg = pch4 + ph2o + ph2 + pco2

    # Qg = np.max(np.array([kp * (Pg-Patm), 0.]))
    Qg = kp * (Pg-Patm) * Pg/Patm
    pT8  = KLa * (S_h2  - 16*KHh2*ph2)
    pT9  = KLa * (S_ch4 - 64*KHch4*pch4)
    pT10 = KLa * (S_co2 - KHco2*pco2) 

    # Inhibitor contributions(I5-I12)
    Iaa = 10**(-(3/(pH_UL_aa - pH_LL_aa))*(pH_LL_aa+pH_UL_aa)/2) / (S_H_i**(3/(pH_UL_aa - pH_LL_aa)) + 10**(-(3/(pH_UL_aa - pH_LL_aa))*(pH_LL_aa+pH_UL_aa)/2)); 
    Iac = 10**(-(3/(pH_UL_ac - pH_LL_ac))*(pH_LL_ac+pH_UL_ac)/2) / (S_H_i**(3/(pH_UL_ac - pH_LL_ac)) + 10**(-(3/(pH_UL_ac - pH_LL_ac))*(pH_LL_ac+pH_UL_ac)/2)); 
    Ih2 = 10**(-(3/(pH_UL_h2 - pH_LL_h2))*(pH_LL_h2+pH_UL_h2)/2) / (S_H_i**(3/(pH_UL_h2 - pH_LL_h2)) + 10**(-(3/(pH_UL_h2 - pH_LL_h2))*(pH_LL_h2+pH_UL_h2)/2)); 

    I5  = 1 / (1 + (KS_IN/S_IN)) * Iaa
    I6  = I5
    I7  = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_h2/K_Ih2fa)) * Iaa
    I8  = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_h2/K_Ih2c4)) * Iaa
    I9  = I8
    I10 = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_h2/K_Ih2pro)) * Iaa
    I11 = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_nh3/K_Inh3)) * Iac
    I12 = 1 / (1 + (KS_IN/S_IN)) * Ih2

    # Particulate Matter
    p1 = kdis  * X     # Disintegration rate
    p2 = kh_ch * X_ch  # Hydrolysis of carbohydrates rate
    p3 = kh_pr * X_pr  # Hydrolysis of proteins rate
    p4 = kh_li * X_li  # Hydrolysis of lipids rate

    # Liquid Phase
    p5  = k_msu * S_su / (K_Ssu + S_su) * X_su * I5 # Uptake of sugars
    p6  = k_maa * S_aa / (K_Saa + S_aa) * X_aa * I6 # Uptake of amino acids
    p7  = k_mfa * S_fa / (K_Sfa + S_fa) * X_fa * I7 # Uptake of long chain fatty acids    
    p8  = k_mc4 * S_va / (K_Sc4 + S_va) * S_va / (S_bu + S_va ) * X_c4 * I8 + 1e-6 # Uptake of valerate + 1e-6
    p9  = k_mc4 * S_bu / (K_Sc4 + S_bu) * S_bu / (S_bu + S_va ) * X_c4 * I9 + 1e-6 # Uptake of butyrate + 1e-6
    p10 = k_mpro* S_pro/ (K_Spro+ S_pro)* X_pro * I10 # Uptake of propionate
    p11 = k_mac * S_ac / (K_Sac + S_ac) * X_ac * I11  # Uptake of acetate
    p12 = k_mh2 * S_h2 / (K_Sh2 + S_h2) * X_h2 * I12  # Uptake of hydrogen gas
    
    # Decay in particulate matter
    p13 = kdec_Xsu * X_su  # Decay of sugars
    p14 = kdec_Xaa * X_aa  # Decay of amino acids
    p15 = kdec_Xfa * X_fa  # Decay of long chain fatty acids
    p16 = kdec_Xc4 * X_c4  # Decay of valerate
    p17 = kdec_Xpro* X_pro # Decay of propionate
    p18 = kdec_Xac * X_ac  # Decay of acetate
    p19 = kdec_Xh2 * X_h2  # Decay of hydrogen gas
    
    # Carbon balance
    s1  = (-1 * C_xc + fsi_xc * C_sI + fch_xc * C_ch + fpr_xc * C_pr + fli_xc * C_li + fxi_xc * C_xI) 
    s2  = (-1 * C_ch + C_su)
    s3  = (-1 * C_pr + C_aa)
    s4  = (-1 * C_li + (1 - ffa_li) * C_su + ffa_li * C_fa)
    s5  = (-1 * C_su + (1 - Ysu) * (fbu_su * C_bu + fpro_su * C_pro + fac_su * C_ac) + Ysu * C_bac)
    s6  = (-1 * C_aa + (1 - Yaa) * (fva_aa * C_va + fbu_aa * C_bu + fpro_aa * C_pro + fac_aa * C_ac) + Yaa * C_bac)
    s7  = (-1 * C_fa + (1 - Yfa) * 0.7 * C_ac + Yfa * C_bac)
    s8  = (-1 * C_va + (1 - Yc4) * 0.54 * C_pro + (1 - Yc4) * 0.31 * C_ac + Yc4 * C_bac)
    s9  = (-1 * C_bu + (1 - Yc4) * 0.8 * C_ac + Yc4 * C_bac)
    s10 = (-1 * C_pro + (1 - Ypro) * 0.57 * C_ac + Ypro * C_bac)
    s11 = (-1 * C_ac + (1 - Yac) * C_ch4 + Yac * C_bac)
    s12 = ((1 - Yh2) * C_ch4 + Yh2 * C_bac)
    s13 = (-1 * C_bac + C_xc) 
    sigma =  (s1 * p1 + s2 * p2 + s3 * p3 + s4 * p4 
              + s5 * p5 + s6 * p6 + s7 * p7 + s8 * p8 
              + s9 * p9 + s10 * p10 + s11 * p11 + s12 * p12 
              + s13 * (p13 + p14 + p15 + p16 + p17 + p18 + p19)
    )

    # Ion rates
    pA4 = k_A_B_va * (S_va_i * (K_a_va + S_H_i) - K_a_va*S_va)
    pA5 = k_A_B_bu * (S_bu_i * (K_a_bu + S_H_i) - K_a_bu*S_bu)
    pA6 = k_A_B_pro * (S_pro_i * (K_a_pro + S_H_i) - K_a_pro*S_pro)
    pA7 = k_A_B_ac * (S_ac_i * (K_a_ac + S_H_i) - K_a_ac*S_ac)
    pA10 = k_A_B_co2 * (S_hco3_i * (K_a_co2 + S_H_i) - K_a_co2*S_IC)
    pA11 = k_A_B_IN * (S_nh3 * (K_a_IN + S_H_i) - K_a_IN*S_IN)

  ## 5. Rate ODEs

    # Particulate Matter
    dXdt = Q/V * (0.000 - X) - p1 + p13+p14+p15+p16+p17+p18+p19 # 13 Disintegration rate
    dX_chdt = Q/V * (Xi_ch - X_ch) + fch_xc*p1 - p2  # 14 Carbohydrates hydrolysis rate
    dX_prdt = Q/V * (Xi_pr - X_pr) + fpr_xc*p1 - p3  # 15 Proteins hydrolysis rate
    dX_lidt = Q/V * (Xi_li - X_li) + fli_xc*p1 - p4  # 16 Lipids hydrolysis rate
    
    # Particulate Matter Continued
    dX_sudt = Q/V * (0.000 - X_su) + Ysu*p5 - p13  # 17 Sugars hydrolysis rate
    dX_aadt = Q/V * (0.000 - X_aa) + Yaa*p6 - p14  # 18 Amino acids hydrolysis rate
    dX_fadt = Q/V * (0.000 - X_fa) + Yfa*p7 - p15  # 19 Fatty acids hydrolysis rate
    dX_c4dt = Q/V * (0.000 - X_fa) + Yc4*p8 + Yc4*p9 - p16  # 20
    dX_prodt= Q/V * (0.000 - X_pro)+ Ypro*p10 - p17  # 21 Propionate hydrolysis rate
    dX_acdt = Q/V * (0.000 - X_ac) + Yac*p11 - p18   # 22 Acetate hydrolysis rate
    dX_h2dt = Q/V * (0.000 - X_h2) + Yh2*p12 - p19   # 23 Hydrogen hydrolysis rate
    
    # Liquid Phase
    dS_sudt = Q/V * (0.000-S_su) + p2 + (1-ffa_li)*p4 - p5 
    dS_aadt = Q/V * (0.000-S_aa) + p3 - p6
    dS_fadt = Q/V * (0.000-S_fa) + ffa_li*p4 - p7
    dS_vadt = Q/V * (0.000-S_va) + (1+Yaa)*fva_aa*p6 - p8 
    dS_budt = Q/V * (0.000-S_bu) + (1-Ysu)*fbu_su*p5 + (1-Yaa)*fbu_aa*p6 - p9 
    dS_prodt= Q/V * (0.000-S_pro)+ (1-Ysu)*fpro_su*p5+ (1-Yaa)*fpro_aa*p6 + (1-Yc4)*0.54*p8 - p10  
    dS_acdt =(Q/V * (0.000-S_ac) + (1-Ysu)*fac_su*p5 + (1-Yaa)*fac_aa*p6
              + (1-Yfa)*0.7*p7 + (1-Yc4)*0.31*p8 + (1-Yc4)*0.8*p9
              + (1-Ypro)*0.57*p10 - p11) 
    dS_h2dt =(Q/V * (0.000-S_h2) + (1-Ysu)*fh2_su*p5 + (1-Yaa)*fh2_aa*p6 
              + (1-Yfa)*0.3*p7 + (1-Yc4)*0.15*p8 + (1-Yc4)*0.2*p9
              + (1-Ypro)*0.43*p10 - p12 - pT8)
    dS_ch4dt= Q/V * (0.000-S_ch4) + (1-Yac)*p11 + (1-Yh2)*p12 - pT9

    # nitorgen inhibitory Phase
    dS_INdt = (Q/V * (0.000-S_IN) 
            - Ysu*Nbac*p5 + (Naa-Yaa*Nbac)*p6 - Yfa*Nbac*p7 - Yc4*Nbac*p8
            - Yc4*Nbac*p9 - Ypro*Nbac*p10 - Yac*Nbac*p11 - Yh2*Nbac*p12
            + (Nbac-Nxc)*(p13+p14+p15+p16+p17+p18+p19)
            + (Nxc - fxi_xc*NI - fsi_xc*NI - fpr_xc*Naa)*p1)
    
    dS_ICdt = Q/V  * (0.000 - S_IC) - sigma - pT10

    # Gas phase odes
    dG_h2dt  = - G_h2  * Qg/Vg + pT8*V/Vg            
    dG_ch4dt = - G_ch4 * Qg/Vg + pT9*V/Vg
    dG_co2dt = - G_co2 * Qg/Vg + pT10*V/Vg

    # Ion phase odes
    dS_cat_idt = Q/V  * (0.000 - S_cat_i)
    dS_an_idt = Q/V  * (0.000 - S_an_i)
    dS_va_idt = -pA4
    dS_bu_idt = -pA5
    dS_pro_idt = -pA6
    dS_ac_idt = -pA7
    dS_hco3_idt = -pA10
    dS_nh3dt = -pA11

  ## 6. Output
    dX_c4dt = dX_c4dt if dX_c4dt>=0 else 1e-6

    dz[0], dz[1], dz[2], dz[3], \
    dz[4], dz[5], dz[6], dz[7], dz[8], dz[9], dz[10], \
    dz[11], dz[12], dz[13], \
    dz[14], dz[15], dz[16], dz[17], dz[18], dz[19], dz[20], dz[21] ,dz[22], dz[23] ,dz[24], \
    dz[25], dz[26], dz[27], dz[28], dz[29] ,dz[30], dz[31] ,dz[32]  =  \
    dXdt, dX_chdt, dX_lidt, dX_prdt, \
    dS_sudt, dS_aadt, dS_fadt, dS_vadt, dS_budt, dS_prodt, dS_acdt, \
    dS_h2dt, dS_ch4dt, dS_INdt, \
    dG_h2dt, dG_ch4dt, dX_sudt, dX_aadt, dX_fadt, dX_c4dt, dX_prodt, dX_acdt, dX_h2dt, dG_co2dt, dS_ICdt,\
    dS_cat_idt, dS_an_idt, dS_va_idt, dS_bu_idt, dS_pro_idt, dS_ac_idt, dS_hco3_idt, dS_nh3dt

funcptr = Model_lsoda.address

@njit
def calc_gas(final_sol):
    R     = 0.083145 # bar.M^-1.K^-1
    Patm  = 1.013    # bar
    T     = 308.15   # K
    kp    = 5e4      # [m3.d-1.bar-1] overhead pressure
    G_h2  = final_sol[:,14]
    G_ch4 = final_sol[:,15]
    G_co2 = final_sol[:,23]
    ph2  = G_h2  * R * T / 16
    pch4 = G_ch4 * R * T / 64
    pco2 = G_co2 * R * T
    ph2o = 0.0557
    Pg = pch4 + ph2o + ph2 + pco2 
    Qg = kp * (Pg-Patm) * Pg/Patm
    Qg[Qg < 0] = 0

    K_w = 2.08e-14
    S_IN = final_sol[:,13]
    S_IC = final_sol[:,24]
    S_cat_i = final_sol[:,25]
    S_an_i = final_sol[:,26]
    S_va_i = final_sol[:,27]
    S_bu_i = final_sol[:,28]
    S_pro_i = final_sol[:,29]
    S_ac_i = final_sol[:,30]
    S_hco3_i = final_sol[:,31]
    S_nh3 = final_sol[:,32]
    S_nh4_i = S_IN - S_nh3; 
    phi = S_cat_i + S_nh4_i - S_hco3_i - S_ac_i/64 - S_pro_i/112 - S_bu_i/160 - S_va_i/208 - S_an_i; 
    S_H_i = -phi*0.5 + 0.5*(phi**2+4*K_w)**0.5; 
    pH = -np.log10(S_H_i); 
    
    return Qg, pH

@njit
def run_stage(u0, deltaT, data):
    t_eval = np.linspace(0, deltaT, int(deltaT*50+1))
    sol, success =  lsoda(funcptr, u0, np.linspace(0,.01,20), np.array(data), rtol=1e-5, atol=1e-8)
    sol, success =  lsoda(funcptr, sol[-1], t_eval, np.array([.0,.0,.0, 1., 1.]), rtol=1e-5, atol=1e-8)
    Qch4, Qco2 = calc_gas(sol)
    Ach4 = np.sum(Qch4) / (deltaT*50)
    Aco2 = np.sum(Qco2) / (deltaT*50) 
    return sol, Qch4, Qco2, Ach4, Aco2

def get_alphas(Alpha_list_i, f, f_tot):
    Alpha1 = Alpha_list_i[0] * f[0]/f_tot
    Alpha2 = Alpha_list_i[1] * f[1]/f_tot
    Alpha3 = Alpha_list_i[2] * f[2]/f_tot
    Alphas = np.zeros(3)
    for z in range(3):
        Alphas[z]=(Alpha1[z]+Alpha2[z]+Alpha3[z])
    return Alphas

deltaT = 1
Q_range =  np.arange(120, 390+30,30) 
state = np.loadtxt('state_2x1.txt', delimiter=',')
u0s = np.loadtxt('u0s_2x1.txt', delimiter=',')
States = np.array(state)
u0s = np.array(u0s)
N = len(States)
action = Q_range / 3 
L = len(action)
Actions=[[a1,a2,a3] for a1 in action for a2 in action for a3 in action] 
fw_kernel = stats.gaussian_kde(fw_data, weights=fw_weight) 
aw_kernel = stats.gaussian_kde(aw_data, weights=aw_weight) 
mw_kernel = stats.gaussian_kde(mw_data, weights=mw_weight) 
M     = 30 
T0    =  3  
T     = 20 + T0 
targets = [150, 110, 200, 170]
c_hat = np.ones(T) * targets[0]; c_hat[5+T0:10+T0] = targets[1]; c_hat[10+T0:15+T0]=targets[2]; c_hat[15+T0:]= targets[3]
total_itr = T*N*L**3*M
Opt_val = np.zeros(shape=(N,T))  
Opt_act = np.zeros(shape=(N,T)) 

def MDP_Loop(T, deltaT, M, N, L, States, Actions, u0s):
    Opt_val = np.zeros(shape=(N,T))  
    Opt_act = np.zeros(shape=(N,T)) 
    u0 = np.array([1.73722099e-02, 1.74298063e-04, 2.61349182e-04, 1.74298063e-04,
                6.82369318e-04, 1.90034584e-04, 1.92398955e-02, 1.62566549e-04,
                2.00606504e-04, 3.27529418e-03, 7.74516357e-03, 3.89307460e-08,
                4.47232494e-02, 3.53516771e-02, 2.09424381e-06, 1.49718568e+00,
                7.64469470e-02, 5.72114733e-02, 5.53006003e-02, 1.16728401e-01,
                1.23954086e-02, 7.85621602e-02, 3.79352960e-02, 1.40217979e-02,
                6.30099368e-02, 3.69686639e-02, 1.84843319e-02, 1.61508352e-04,
                1.99404528e-04, 3.25300282e-03, 7.70510852e-03, 5.32559640e-02,
                4.28445882e-04])
   
    for ti in range(T): 
        t = T - ti - 1
        for k in range(N):
            j = 0 
            Alpha_list = create_rand_dist([fw_kernel,aw_kernel,mw_kernel], L**3*M)
            s = States[k] 
            u0 = u0s[k]
            rec_f = np.zeros(L**3)
            for l in range(L**3): 
                f = Actions[l] 
                f_tot = np.sum(Actions[l,:]) 
                c_ch4 = np.zeros(M) 
                c_obj = np.zeros(M)
                
                for m in range(M):
                    Alphas = get_alphas(Alpha_list[j], f, f_tot)   
                    j += 1
                    sol, Qch4, Qco2, Ach4, Aco2 = run_stage(u0, deltaT, (Alphas[0], Alphas[1], Alphas[2], f_tot, 1.))
                    c_ch4[m] = Ach4
                    c_idx = np.argmin(np.abs(States - np.array([s[1], Ach4])).sum(axis=1))
                    loss = np.abs(Ach4 - c_hat[t])
                    c_obj[m] = loss if t == T-1 else loss + Opt_val[c_idx][t+1]
                rec_f[l] = np.mean(c_obj)    
            Opt_val[k][t] = np.min(rec_f) 
            Opt_act[k][t] = int(np.argmin(rec_f))
        print(Opt_act[t])
    return Opt_val, Opt_act 

np.savetxt('MDP_Opt_act.txt', Opt_act, delimiter=',')
np.savetxt('MDP_Opt_val.txt', Opt_val, delimiter=',')





