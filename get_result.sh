#! /bin/bash
    function read_dir(){
        for file in `ls $1`      
        do
            name=$1"/"$file
            if [ -d $name ]  
            then
                read_dir $name
            else
                
                if [[ "$name" == */train_log.txt ]]
                then
                    echo $name
                    #tail -n 2 $name | head -n 1
                    grep best_ $name | tail -c 20
                fi
            fi
        done
    }   

    read_dir $1
