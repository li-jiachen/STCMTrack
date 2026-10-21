from collections import namedtuple







LabelCp = namedtuple( 'LabelCp' , [

    'name'        , 
                    

    'id'          , 
                    

    'hasInstances', 

    'ignoreInEval', 
                    

    'color'       , 
    ] )










labelsCp = [
    
    LabelCp(  'ignore'               ,  0 , False        , True         , (250,170, 30) ),
    LabelCp(  'pedestrian'           ,  1 , True         , False        , (220, 20, 60) ),
    LabelCp(  'rider'                ,  2 , True         , False        , (  0,  0,142) ),
    LabelCp(  'sitting person'       ,  3 , True         , False        , (107,142, 35) ),
    LabelCp(  'person (other)'       ,  4 , True         , False        , (190,153,153) ),
    LabelCp(  'person group'         ,  5 , False        , True         , (255,  0,  0) ),
]









name2labelCp      = { label.name    : label for label in labelsCp }

id2labelCp        = { label.id      : label for label in labelsCp }
