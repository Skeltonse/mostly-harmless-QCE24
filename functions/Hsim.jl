using DataFrames
using CSV
using SpecialFunctions

####specify inputs (maybe eventually depreciate)
epsi=10^(-14) #to match Dong et Al. 21'
t=1500
filename="hsim_coeffs_epsi_" * string(epsi) * "_t_" * string(t) * ".csv"
ifsave=true

#####generate coefficients
function asymp_scaling(t, epsi)
    #begin by using (78) in grand unifications , lemma 59 in Gilyén
    r=abs(t)+log(1/epsi)/log(exp(1)+log(1/epsi)/abs(t))
    return r
end

function k_select(t, epsi)
    #take the martyn scaling at first...circle back to understanding the Gily#en paper whe you do this for real
    r=asymp_scaling(exp(1)/2*abs(t), 5/4*epsi)
    return Int(floor(r)/2)
end

function k_select_Dong(t, epsi)
    #take the Dong scaling
    #r=asymp_scaling(exp(1)/2*abs(t), 5/4*epsi)
    r=exp(1)*abs(t)/2+log10(1/epsi)
    return Int(ceil(r))
end

function hsim_coeffs(t, epsi, prefact=1/2)
    k=k_select_Dong(t, epsi)

    n=Int(2*k+1)

    cheby_even=zeros(Int(2*k+2))
    cheby_odd=zeros(Int(2*k+2))

    cheby_even[1]=prefact*besselj0(t)
    ###build accoridng to (76, 77) in Martyn, may be able to clean up later
    #but forget the factor of 2 bc we just divide it to get the complex coeffs anyways
    for l in range(0, k)
        cheby_even[2*l+1]=prefact*2*(-1)^l*besselj(2*l, t)
        cheby_odd[2*l+2]=prefact*2*(-1)^l*besselj(2*l+1, t)
    end
    recip=vcat(reverse(cheby_even[2:end]), vcat(2*cheby_even[1], cheby_even[2:end]))/2
    antirecip=vcat(reverse(cheby_odd[2:end]), vcat(2*cheby_odd[1],cheby_odd[2:end]))/2
    
    return recip, antirecip, n
    #return cheby_even, 1im*cheby_odd, n
end



ceven, codd, n=hsim_coeffs(t, epsi)
println(n)
####save data as a csv file
#current_dir=pwd() wrong directory, since visual studios actually runs things in a differnet path location
current_file_path = @__FILE__
save_path=replace(replace(current_file_path, "functions" => "csv_files"), "Hsim.jl" => filename)

df= DataFrame(coscoeff=ceven, sincoeff=codd, highest_deg=n, epsilon=epsi, evoltime=t)
if ifsave==true
    CSV.write(save_path, df)
end

