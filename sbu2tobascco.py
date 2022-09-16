#!/usr/bin/env python3

import os
import re
import glob
import argparse
import traceback
import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.core.sites import Site


code_desc=""" 
Module to convert SBU .xyz files to a formate compatible with ToBasCCo.

Converts 'X' type connection points to SBU-Xe-Rn format (ToBasCCo).

Reads modified .xyz format used in "pormake" SBUs and creates a .mol file
compatiable with ToBasCCo.
"""

parser = argparse.ArgumentParser(description=code_desc)
parser.add_argument('search_path', 
                    help='path where the sbu .xyz files are located')
args=parser.parse_args()


def parse_xyz(xyz_file):
    '''Read all atom & connection point info from SBU .xyz file'''
    pos_data = []
    bond_table = []
    with open(xyz_file, 'r') as xyzf:
        xyz_lines = xyzf.read().split("\n")
    no_atoms = int(xyz_lines[0].strip())
    cp_ind = xyz_lines[1].split()

    atom_xyz_patt = '[ \t]*[a-zA-Z *]{1,2}[ \t]*[-]*[\d]*\.\d*[ \t]*[-]*[\d]*\.\d*[ \t]*[-]*[\d]*\.\d*[ \t]*'
    bond_patt = '^[ \t]*[\d]+[ \t]*[\d]+[ \t]+[\w][ \t]*$'
    for line in xyz_lines[2:no_atoms+2]:
        if re.match(atom_xyz_patt, line):
            pos_data.append(line)
        else:
            print(f'{xyz_file} >>> POS != expectation >>> CHK format!!!')
    for line in xyz_lines[no_atoms+2:]:
        if re.match(bond_patt, line):
            bond_table.append(line)
        else:
            continue

    return no_atoms, cp_ind, pos_data, bond_table

def fix_cp_pos(cp_atom_ind, oth_atom_ind, pos_list, bond_list):
    '''Converts 'X' connection point format to a Xe-Rn format (tobascco)'''
    # Reads 'X' & connected atom's position lines from list
    cp_line = pos_list[int(cp_atom_ind)].split()
    oth_line = pos_list[int(oth_atom_ind)].split()
    # Collects 'X' & connected atom's positions into arrays
    cp_pos = np.array([float(cp_line[1]),
                       float(cp_line[2]),
                       float(cp_line[3])])
    other_pos = np.array([float(oth_line[1]),
                          float(oth_line[2]),
                          float(oth_line[3])])
    # Calculates vector pointing towards 'X' atom
    cp_vector = cp_pos - other_pos
    # Calculates the position of the Rn atom relative to 'X' position 
    new_cp_pos = cp_pos + (0.5*cp_vector)
    # Rounds x,y,z positions to conventional no. decimals
    for i in range(len(new_cp_pos)):
        new_cp_pos[i] = round(new_cp_pos[i], 4)
    # Prepare new pos_list entries for Xe(@ 'X' pos) & Rn(@ new_cp_pos)
    new_cp_line_1 = f'Xe   {cp_pos[0]} {cp_pos[1]} {cp_pos[2]}'
    new_cp_line_2 = f'Rn   {new_cp_pos[0]} {new_cp_pos[1]} {new_cp_pos[2]}'
    # new_cp_line_1 = f'Xe   {cp_pos[0]} {cp_pos[1]} {cp_pos[2]}'
    # new_cp_line_2 = f'Rn   {new_cp_pos[0]} {new_cp_pos[1]} {new_cp_pos[2]}' 
    new_bond_line = f' {cp_atom_ind} {len(pos_list)} S'
    # Update pos_list & bond_list with new atoms/bonds
    pos_list[int(cp_atom_ind)] = new_cp_line_1
    pos_list.append(new_cp_line_2)
    bond_list.append(new_bond_line)
    return 0

def center_pos(pos_list):
    '''Centers the SBU position using pymatgen (fix pos from SBU mining)'''
    center_pos_list = []
    sites = []
    coords = []
    for pos in pos_list:
        s_pos = pos.split()
        if s_pos[0] != 'X':
            coords.append([float(s_pos[1]), float(s_pos[2]), float(s_pos[3])])
            sites.append(s_pos[0])
    
    mol = Molecule(sites, coords)
    center = mol.center_of_mass

    for pos in pos_list:
        s_pos = pos.split()
        new_x = round(float(s_pos[1])-center[0], 4)
        new_y = round(float(s_pos[2])-center[1], 4)
        new_z = round(float(s_pos[3])-center[2], 4)
        new_pos_line = f'{s_pos[0]}   {new_x} {new_y} {new_z}'
        center_pos_list.append(new_pos_line)

    return center_pos_list

def convert_cp(atom_ct, conn_points, pos, bonds):
    '''Finds connection points in the bond table to convert'''
    # Searches bonds for conn_points
    for bond in bonds[0:len(bonds)]:
        spl_l = bond.strip().split()
        # Typical case for bond_table data
        if len(spl_l) == 3:
            atom1 = spl_l[0]
            atom2 = spl_l[1]
            if atom1 in conn_points:
                fix_cp_pos(atom1, atom2, pos, bonds)
                atom_ct += 1
            elif atom2 in conn_points:
                fix_cp_pos(atom2, atom1, pos, bonds)
                atom_ct += 1
            else:
                continue
        # Handles case where large atom_indices merge together in bond_table
        elif len(spl_l) == 2:
            # Separate merged atom_indices
            len_offset = len(spl_l[0])-3
            atom1 = spl_l[0][0:len_offset]
            atom2 = spl_l[0][len_offset:6]
            if atom1 in conn_points:
                fix_cp_pos(atom1, atom2, pos, bonds)
                atom_ct += 1
            elif atom2 in conn_points:
                fix_cp_pos(atom2, atom1, pos, bonds)
                atom_ct += 1
            else:
                continue

    return atom_ct, conn_points, pos, bonds

def format_mol_line(xyz_pos_lines, xyz_bond_lines):
    '''Format .mol text according to convention'''
    mol_pos_lines = []
    mol_bond_lines = []
    chk_dup_bonds = []
    # Format each .xyz line to its corresponding .mol line
    for line in xyz_pos_lines:
        spl_l = line.split()
        x_val = ' ' * (10-len(str(spl_l[1]))) + (str(spl_l[1]))
        y_val = ' ' * (10-len(str(spl_l[2]))) + (str(spl_l[2]))
        z_val = ' ' * (10-len(str(spl_l[3]))) + (str(spl_l[3]))
        buffer = ' ' * (7- len(str(spl_l[0])))
        new_l = f'{x_val}{y_val}{z_val} {spl_l[0]}{buffer}0' 
        mol_pos_lines.append(new_l)
    # Convert bond typing to accepted format
    bond_type_conv = {'S': '1',
                      'D': '2',
                      'T': '3',
                      'A': '4'}
    for line in xyz_bond_lines:
        spl_l = line.split()
        bond_type = bond_type_conv[spl_l[-1]]
        # Remove duplicate bonds (issue resolved, but may still emerge)
        if sorted([spl_l[0], spl_l[1]]) not in chk_dup_bonds:
            atom_mash = ''
            for at in [spl_l[0], spl_l[1]]:
                at = str(int(at) + 1)
                if len(at) == 2:
                    atom_mash += f' {at}'
                elif len(at) == 1:
                    atom_mash += f'  {at}'
                elif len(at) == 3:
                    atom_mash += f'{at}'
            new_l = f'{atom_mash}  {bond_type}' 
            mol_bond_lines.append(new_l)
            chk_dup_bonds.append(sorted([spl_l[0], spl_l[1]]))
        else:
            # print('Found duplicate bond')
            continue
    return mol_pos_lines, mol_bond_lines

def write_mol(atom_ct, conn_points, pos, bonds, filename):
    '''Format the generated .mol file with expected information'''
    mol_name = os.path.basename(filename).replace('.xyz', '.mol')
    write_path = os.path.dirname(filename) + '/test_mol_files/'
    full_mol = os.path.join(write_path, mol_name)
    with open(full_mol, 'w') as molf:
        name = f'{mol_name[0:-4]}\n'
        fix_atom_ct = ' ' * (3-len(str(atom_ct))) + f'{atom_ct}'
        fix_bonds = ' ' * (3-len(str(len(bonds)))) + f'{len(bonds)}'
        x = f'{fix_atom_ct}{fix_bonds}  0  0  0  0  0  0  0  0999 V2000\n'
        pos_text = '\n'.join(pos)
        bond_text = '\n'.join(bonds)
        molf.write(name)
        molf.write('\n'*2)
        molf.write(x)
        molf.write(f"{pos_text}\n")
        molf.write(f"{bond_text}\n")
        molf.write('M  END')


def main():
    # Search for SBU .xyz files in search_path
    files = glob.glob(args.search_path + '/*.xyz', recursive=False)
    os.mkdir(args.search_path + "/tobascco_mol_files")
    # Files where no 'X' are found
    no_cp = []
    # Any other errors during parsing & X-conversion
    errors = []
    for file in files:
        try:    
            ct, cp, pos, bonds = parse_xyz(file)
            centered_pos = center_pos(pos)
            fix_ct, fix_cp, fix_pos, fix_bonds = convert_cp(ct,
                                                            cp,
                                                            centered_pos,
                                                            bonds)
            fix_pos_2, fix_bonds_2 = format_mol_line(fix_pos,
                                                     fix_bonds)
            if fix_cp:
                write_mol(fix_ct, fix_cp, fix_pos_2, fix_bonds_2, file)
            # Case where original xyz files have no 'X' 
            else:
                no_cp.append(file)
                print(f'{file} >> ERROR >> No Connection points found')
        except Exception as e:
            errors.append(os.path.basename(file))
            print(f'{file}>>ERROR>>{e}')
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()


