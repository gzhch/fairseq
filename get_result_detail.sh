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
                    python get_result.py $name
                fi
            fi
        done
    }   

    read_dir $1
