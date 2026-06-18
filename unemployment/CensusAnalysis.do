global PROJECT_ROOT "."
global UNEMPLOYMENT_DIR "$PROJECT_ROOT/unemployment"

set seed 20260618

use "$UNEMPLOYMENT_DIR/usa_00203-001.dta", clear
drop if slwt == 0
g census_u = empstat==2
drop if empstat== 3 | empstat == 0
g rand = runiform(0,1)
bysort year (educd rand): g ed_rank = _n
g all_tercile = . 
forvalues yr = 1940(10)2010 {
xtile tercile`yr' = ed_rank  if year == `yr' [fw = perwt], nquantile(3) 
replace all_tercile = tercile`yr' if year == `yr'
}

xtile tercile2019 = ed_rank  if year == 2019 [fw = perwt], nquantile(3) 
replace all_tercile = tercile2019 if year == 2019 
xtile tercile2024 = ed_rank  if year == 2024 [fw = perwt], nquantile(3) 
replace all_tercile = tercile2024 if year == 2024

collapse(mean) census_u [fw = perwt], by(year all_tercile)

reshape wide census_u , i(year) j(all_tercile)
twoway (line census_u1 year) (line census_u2 year) (line census_u3 year)
sort year 

tempfile census_unemployment
save `census_unemployment', replace



use "$UNEMPLOYMENT_DIR/cps_00180.dta", clear
keep if labforce == 2
g cps_u = inlist(empstat,20,21,22)
g rand = runiform(0,1)
drop if asecwt<=0
bysort year (educ rand): g ed_rank = _n


g all_tercile = . 
forvalues yr = 1962(1)2025 {
xtile tercile`yr' = ed_rank  if year == `yr' [fw = round(asecwt,1)], nquantile(3) 
replace all_tercile = tercile`yr' if year == `yr'
}
drop if year == 1963

g count = 1 
collapse(mean) cps_u (sum) count [fw = round(asecwt,1)], by(year all_tercile)

reshape wide cps_u count, i(year) j(all_tercile)

format count* %12.0f

twoway (line cps_u1 year) (line cps_u2 year) (line cps_u3 year) 

sort year 
merge 1:1 year using `census_unemployment'


sort year
twoway (line cps_u1 year) (line cps_u2 year) (line cps_u3 year) (line census_u1 year) (line census_u2 year)  (line census_u3 year) 
twoway (line cps_u1 year) (line cps_u3 year) (line census_u1 year) (line census_u3 year) 
