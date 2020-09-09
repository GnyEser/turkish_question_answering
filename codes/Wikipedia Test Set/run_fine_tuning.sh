for i in {0..17}
do
	for j in {0..2}
	do
		for k in {0..2}
		do
			python question_answering_fine_tuning.py $i $j $k
			sleep 90
		done
	done
done
