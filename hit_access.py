ROOT_FILENAME = "mcv5.1.mupage_10G.km3new_AAv1.jterbr00008494.3350.root"

import km3pipe as kp
#aaaa
pump = kp.io.jpp.TimeslicePump(filename=ROOT_FILENAME, stream='SN')
for blob in pump:
    hits = blob['TSHits']
