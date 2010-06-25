#!/usr/bin/env perl
use warnings;
$flag = -1;
while(<>){
	chomp;

	if(/Options: (.*)$/){
		$Options=$1;
	}
	if(/Correctly Classified Instances([^0-9]*[0-9]*[^0-9]*)(.*)$/)
		{
			if($flag == 1) 
				{
				$Correct = $2; 
				print"$Options $Correct \n";
				$Options = "-1";
				$Correct = "-1";
				}
			$flag *= -1;
		}
}

