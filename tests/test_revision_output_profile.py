import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import pytest
from profile_revision_outputs import distribution_profile, loss_windows, fragment_category
from report_revision_output_profile import contrast, describe


def test_signed_kl_accounting_and_mass_transfer():
    p=np.array([.7,.2,.1]);q=np.array([.2,.4,.4])
    _,_,terms,tv=distribution_profile(np.log(p),np.log(q))
    assert terms[0]>0 and (terms[1:]<0).all()
    assert tv==pytest.approx(.5)
    assert terms.sum()==pytest.approx(.7*np.log(3.5)+.2*np.log(.5)+.1*np.log(.25))
    with pytest.raises(AssertionError):distribution_profile(np.log(p*2),np.log(q))


def test_nonoverlapping_windows_and_question_weighting():
    short=loss_windows([4.,2.]); assert short['after16'] is None and short['first16']==3
    long=loss_windows([1.]*16+[5.]*84)
    assert long['after16']==5 and long['full']==4.36
    # Questions, not their unequal token counts, are the statistical units.
    assert describe([short['full'],long['full']])['mean']==pytest.approx(3.68)


def test_pairing_uses_ids_and_rejects_missing_or_duplicate_questions():
    f=pd.DataFrame([{'sample_id':sid,'method':m,'width':16,'metric':v} for sid,m,v in
        [('b','a',4),('a','a',9),('a','b',3),('b','b',2)]])
    assert contrast(f,'a',16,'b',16,'metric').tolist()==[6,2]
    with pytest.raises(AssertionError):contrast(f.iloc[:3],'a',16,'b',16,'metric')
    with pytest.raises(AssertionError):contrast(pd.concat([f,f.iloc[:1]]),'a',16,'b',16,'metric')


def test_fragment_rules_do_not_guess_semantics():
    assert fragment_category(' 12')=='number_fragment'
    assert fragment_category(' plus')=='alphabetic_fragment'
    assert fragment_category('=')=='operator_fragment'
    assert fragment_category('\\frac')=='latex_or_dollar_fragment'
    assert fragment_category('')=='whitespace'
